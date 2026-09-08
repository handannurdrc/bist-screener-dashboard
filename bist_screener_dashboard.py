"""
BIST 100 Hisse Tarama ve Analiz Dashboard'u
=============================================
Streamlit tabanlı, nesne yönelimli (OOP) mimariye sahip profesyonel
Borsa İstanbul (BIST 100) hisse tarama ve teknik analiz uygulaması.

Çalıştırmak için:
    pip install streamlit yfinance pandas numpy plotly matplotlib
    streamlit run bist_screener_dashboard.py

Not: `matplotlib`, tablodaki renkli gradyan (background_gradient) için
pandas Styler tarafından dahili olarak kullanılır; kurulu değilse kod
otomatik olarak gradyansız (düz) tabloya geri döner.

Not: yfinance, Yahoo Finance üzerinden veri çeker; BIST hisseleri ".IS"
uzantısı ile sorgulanır (ör. "THYAO.IS").

KISA VADELİ PATLAMA TESPİTİ İÇİN EKLENEN GELİŞMİŞ MODÜLLER:
  * ATR (Average True Range) tabanlı dinamik stop-loss.
  * RSI momentum eğimi (RSI'nin son birkaç gündeki yönü).
  * Çoklu gün hacim teyidi (tek günlük hacim gürültüsünü azaltmak için).
  * Bollinger Bant Genişliği ile volatilite sıkışması (squeeze) tespiti.
  * XU100 endeksine göre relatif güç (göreceli performans).
  * Backtester: Stratejinin geçmişte ne sıklıkla ve ne kadar isabetli
    sinyal ürettiğini ölçen geriye dönük test motoru.
  * SignalHistoryStore: Üretilen sinyalleri diske kaydedip zamanla
    gerçekleşen getirileri geriye dönük dolduran kalıcı sinyal günlüğü.

Bu gelişmiş filtrelerin tamamı VARSAYILAN OLARAK KAPALIDIR; orijinal 5
temel kriter (hacim patlaması, likidite, RSI bandı, trend, tavan engeli)
değişmeden çalışmaya devam eder. Yeni göstergeler bilgi amaçlı tabloya
eklenir; kullanıcı "Gelişmiş Filtreler" panelinden isterse bunları da
zorunlu kritere çevirebilir. Bunun nedeni: bir filtrenin gerçekten işe
yarayıp yaramadığını ancak Backtest bölümüyle ölçüp karar vermeniz
tavsiye edilir.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# LOGLAMA YAPILANDIRMASI
# ---------------------------------------------------------------------------
logging.basicConfig(
    filename="screener.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger("bist_screener")


# ---------------------------------------------------------------------------
# 1. YAPILANDIRMA SINIFI (CONFIGURATION)
# ---------------------------------------------------------------------------
@dataclass
class ScreeningConfig:
    """
    Tarama stratejisinde kullanılan tüm ayarlanabilir parametreleri
    tek bir merkezi noktada toplayan yapılandırma sınıfı.

    Kod içine "sihirli sayı" (magic number) gömmek yerine tüm eşik
    değerleri buradan yönetilir; böylece sidebar üzerinden kullanıcı
    tarafından anlık değiştirilebilir hale gelir.
    """

    # RSI (Göreceli Güç Endeksi) bantları
    RSI_MIN: float = 50.0
    RSI_MAX: float = 67.0
    RSI_PERIOD: int = 14

    # Hacim çarpanı: Son hacim, 20 günlük ortalama hacmin kaç katı olmalı
    VOL_MULTIPLIER: float = 1.8
    VOLUME_SMA_PERIOD: int = 20

    # Minimum günlük TL işlem hacmi (likidite filtresi)
    MIN_TRY_VOLUME: float = 50_000_000.0

    # Stop-Loss yüzdesi (ör. 0.02 -> %2)
    STOP_LOSS_PCT: float = 0.02

    # Trend filtreleri için hareketli ortalama periyotları
    SMA_SHORT_PERIOD: int = 20
    SMA_LONG_PERIOD: int = 50

    # Tavan engeli filtresi: dünkü kapanış, dünkü en yüksekten en fazla
    # bu oranda uzak olmalı (ör. 0.985 -> en yüksekten %1.5 uzaklık)
    CEILING_GUARD_RATIO: float = 0.985

    # Veri çekimi için minimum gün sayısı (yeni halka arzları elemek için)
    MIN_HISTORY_DAYS: int = 50

    # Grafik için gösterilecek son gün sayısı
    CHART_LOOKBACK_DAYS: int = 60

    # Yahoo Finance'ten çekilecek geçmiş veri periyodu
    DOWNLOAD_PERIOD: str = "6mo"
    DOWNLOAD_INTERVAL: str = "1d"

    # ---------------------------------------------------------------
    # GELİŞMİŞ MODÜL PARAMETRELERİ (kısa vadeli patlama tespiti için)
    # ---------------------------------------------------------------

    # ATR (Average True Range) tabanlı dinamik stop-loss
    ATR_PERIOD: int = 14
    ATR_STOP_MULTIPLIER: float = 1.5

    # RSI momentum eğimi: son kaç günün RSI farkına bakılacak
    RSI_SLOPE_LOOKBACK: int = 3
    REQUIRE_RSI_RISING: bool = False  # varsayılan KAPALI (geriye dönük uyum)

    # Çoklu gün hacim teyidi: tek günlük hacim gürültüsünü azaltmak için
    # son N günün ortalama hacmi de taban ortalamanın X katı olmalı
    VOLUME_CONFIRM_DAYS: int = 3
    VOLUME_CONFIRM_MULTIPLIER: float = 1.3
    REQUIRE_VOLUME_CONFIRMATION: bool = False  # varsayılan KAPALI

    # Volatilite sıkışması (Bollinger Bant Genişliği daralması)
    BB_PERIOD: int = 20
    BB_STD_MULTIPLIER: float = 2.0
    SQUEEZE_LOOKBACK: int = 100
    SQUEEZE_PERCENTILE_MAX: float = 0.25  # en dar %25'lik dilimde olmalı
    REQUIRE_VOLATILITY_SQUEEZE: bool = False  # varsayılan KAPALI

    # Relatif güç: XU100 endeksine göre göreceli performans
    BENCHMARK_TICKER: str = "XU100.IS"
    RELATIVE_STRENGTH_PERIOD: int = 10
    REQUIRE_POSITIVE_RELATIVE_STRENGTH: bool = False  # varsayılan KAPALI

    # Backtest: sinyalden sonraki kaç günlük getiriye bakılacak
    BACKTEST_FORWARD_DAYS: int = 5

    # Sinyal geçmişi kaydı için dosya yolu
    SIGNAL_HISTORY_FILE: str = "signal_history.csv"


# ---------------------------------------------------------------------------
# 2. VERİ SAĞLAYICI KATMANI (DATA PROVIDER) - Kalıtım (Inheritance) Yapısı
# ---------------------------------------------------------------------------
class BaseDataProvider(ABC):
    """
    Soyut temel sınıf (Abstract Base Class).

    Gelecekte farklı piyasalar (ör. ABD borsaları - NASDAQ/NYSE) için
    yeni sağlayıcı sınıfları (ör. UsDataProvider) bu sınıftan türetilerek
    kolayca eklenebilir. Her alt sınıf kendi hisse listesini ve veri
    çekme mantığını tanımlamak zorundadır.
    """

    @property
    @abstractmethod
    def tickers(self) -> List[str]:
        """İlgili piyasanın hisse senedi sembol listesini döndürür."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def fetch_data(
        tickers: Tuple[str, ...], period: str, interval: str, min_history_days: int
    ) -> Dict[str, pd.DataFrame]:
        """Verilen semboller için OHLCV verisini çeker."""
        raise NotImplementedError


class BistDataProvider(BaseDataProvider):
    """Borsa İstanbul (BIST 100) hisseleri için somut veri sağlayıcı sınıfı."""

    # KRİTİK: BIST 100 endeksini oluşturan hisselerin TAMAMI (Yahoo Finance
    # sembol formatında, ".IS" uzantılı). Liste kısaltılmadan, eksiksiz
    # şekilde yazılmıştır. Endeks içeriği zaman zaman revize edildiğinden
    # (üç ayda bir), gerçek zamanlı kullanımdan önce güncel BIST 100
    # listesiyle karşılaştırılması tavsiye edilir.
    BIST100_TICKERS: List[str] = [
        "AEFES.IS", "AGHOL.IS", "AKBNK.IS", "AKSA.IS", "AKSEN.IS",
        "ALARK.IS", "ALFAS.IS", "ALTNY.IS", "ANSGR.IS", "ARCLK.IS",
        "ASELS.IS", "ASTOR.IS", "AYDEM.IS", "BERA.IS", "BFREN.IS",
        "BIENY.IS", "BIMAS.IS", "BRSAN.IS", "BRYAT.IS", "BSOKE.IS",
        "BTCIM.IS", "BUCIM.IS", "CANTE.IS", "CCOLA.IS", "CIMSA.IS",
        "CLEBI.IS", "CWENE.IS", "DOAS.IS", "DOHOL.IS", "ECILC.IS",
        "ECZYT.IS", "EGEEN.IS", "EKGYO.IS", "ENERY.IS", "ENJSA.IS",
        "ENKAI.IS", "EREGL.IS", "EUPWR.IS", "FENER.IS", "FROTO.IS",
        "GARAN.IS", "GESAN.IS", "GLYHO.IS", "GSRAY.IS", "GUBRF.IS",
        "HALKB.IS", "HEKTS.IS", "IEYHO.IS", "IPEKE.IS", "ISCTR.IS",
        "ISDMR.IS", "ISGYO.IS", "ISMEN.IS", "KARSN.IS", "KAYSE.IS",
        "KCAER.IS", "KCHOL.IS", "KMPUR.IS", "KONTR.IS", "KONYA.IS",
        "KOZAA.IS", "KOZAL.IS", "KRDMD.IS", "LMKDC.IS", "MAGEN.IS",
        "MAVI.IS", "MGROS.IS", "MPARK.IS", "ODAS.IS", "OTKAR.IS",
        "OYAKC.IS", "PEKGY.IS", "PENTA.IS", "PETKM.IS", "PGSUS.IS",
        "RGYAS.IS", "SAHOL.IS", "SASA.IS", "SDTTR.IS", "SISE.IS",
        "SKBNK.IS", "SOKM.IS", "TAVHL.IS", "TCELL.IS", "THYAO.IS",
        "TKFEN.IS", "TOASO.IS", "TSKB.IS", "TTKOM.IS", "TTRAK.IS",
        "TUKAS.IS", "TUPRS.IS", "TURSG.IS", "ULKER.IS", "VAKBN.IS",
        "VESBE.IS", "VESTL.IS", "YEOTK.IS", "YKBNK.IS", "ZOREN.IS",
    ]

    @property
    def tickers(self) -> List[str]:
        """BIST 100 hisse senedi sembol listesini döndürür."""
        return self.BIST100_TICKERS

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_data(
        tickers: Tuple[str, ...],
        period: str = "6mo",
        interval: str = "1d",
        min_history_days: int = 50,
    ) -> Dict[str, pd.DataFrame]:
        """
        Verilen hisse senetleri için toplu (batch) OHLCV verisini çeker.

        Streamlit'te `UnhashableParamError` hatası almamak için:
          * Fonksiyon `@staticmethod` olarak tanımlandı (self parametresi yok).
          * `tickers` parametresi hashlenebilir olması için `tuple` olarak
            gönderiliyor (liste değil).
          * `min_history_days` bilerek `ScreeningConfig` nesnesinin tamamı
            yerine tek bir `int` olarak geçiriliyor: `ScreeningConfig` bir
            `@dataclass` olduğu için hash'lenemez (unhashable) ve doğrudan
            cache'lenen bir fonksiyona parametre olarak verilirse yine
            `UnhashableParamError` hatasına yol açar.
          * `@st.cache_data(ttl=3600)` ile sonuçlar 1 saat boyunca
            önbelleğe alınıyor, gereksiz API çağrıları engelleniyor.

        Not: Bu fonksiyon aynı zamanda XU100 gibi tekil bir endeks
        sembolü (benchmark) için de kullanılabilir; sadece `tickers`
        parametresine tek elemanlı bir tuple geçirmek yeterlidir.

        Dönüş: {ticker: dataframe} şeklinde bir sözlük. Her dataframe
        düzleştirilmiş (Open, High, Low, Close, Volume) sütunlarına sahiptir.
        """
        result: Dict[str, pd.DataFrame] = {}

        try:
            raw_data = yf.download(
                list(tickers),
                period=period,
                interval=interval,
                group_by="ticker",
                threads=True,
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:  # Toplu indirme tamamen başarısız olursa
            logger.error(f"Toplu veri indirme başarısız oldu: {exc}")
            st.error("Veri sağlayıcısına (Yahoo Finance) ulaşılamadı. Lütfen daha sonra tekrar deneyin.")
            return result

        for ticker in tickers:
            try:
                # MultiIndex sütun yapısını her hisse için düzleştir
                if isinstance(raw_data.columns, pd.MultiIndex):
                    if ticker not in raw_data.columns.get_level_values(0):
                        logger.warning(f"{ticker}: Veri kümesinde bulunamadı, atlanıyor.")
                        continue
                    df = raw_data[ticker].copy()
                else:
                    # Tek hisse indirilmişse (MultiIndex oluşmaz)
                    df = raw_data.copy()

                # Tamamen boş satırları temizle
                df = df.dropna(how="all")

                # Kritik sütunlarda NaN içeren satırları at
                required_cols = ["Open", "High", "Low", "Close", "Volume"]
                missing_cols = [c for c in required_cols if c not in df.columns]
                if missing_cols:
                    logger.warning(f"{ticker}: Eksik sütunlar {missing_cols}, atlanıyor.")
                    continue

                df = df.dropna(subset=required_cols)

                # En az `min_history_days` günlük verisi olmayan (yeni halka
                # arz vb.) hisseleri güvenli şekilde ele. Değer kod içine
                # gömülmez; ScreeningConfig.MIN_HISTORY_DAYS üzerinden gelir.
                if len(df) < min_history_days:
                    logger.info(f"{ticker}: Yetersiz geçmiş veri ({len(df)} gün), atlanıyor.")
                    continue

                # Tamamlanmamış canlı seans mumunu filtrele:
                # Son satırın tarihi bugünse ve piyasa henüz kapanmadıysa
                # (ör. index'in son elemanı bugünün tarihiyle aynıysa ve
                # veri seti "canlı" bir şekilde güncelleniyorsa) o satırı at.
                # yfinance sürümüne göre index bazen tz-aware (borsa saat
                # dilimine göre yerelleştirilmiş) dönebilir; bu durumda
                # tz-naive `today` ile doğrudan karşılaştırma
                # "Cannot compare tz-naive and tz-aware timestamps" hatası
                # fırlatır. Bu yüzden karşılaştırmadan önce tz bilgisi
                # güvenli şekilde kaldırılır (tz-naive'e indirgenir).
                last_ts = pd.Timestamp(df.index[-1])
                if last_ts.tzinfo is not None:
                    last_ts = last_ts.tz_localize(None)
                today = pd.Timestamp(datetime.now().date())
                if last_ts.normalize() == today:
                    df = df.iloc[:-1]

                if len(df) < min_history_days:
                    logger.info(f"{ticker}: Canlı mum filtrelendikten sonra yetersiz veri, atlanıyor.")
                    continue

                result[ticker] = df

            except Exception as exc:
                logger.error(f"{ticker}: Veri işlenirken hata oluştu -> {exc}")
                continue

        return result


# ---------------------------------------------------------------------------
# 3. TEKNİK ANALİZ MOTORU (saf pandas & numpy, harici TA kütüphanesi yok)
# ---------------------------------------------------------------------------
class TechnicalAnalyzer:
    """
    Teknik göstergeleri harici bir kütüphaneye (ta-lib, pandas-ta vb.)
    ihtiyaç duymadan, saf pandas ve numpy kullanarak hesaplayan sınıf.
    """

    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        Wilder's Smoothing (Wilder'ın Düzeltme Yöntemi) ile RSI hesaplar.

        Klasik basit hareketli ortalama yerine, Welles Wilder'ın orijinal
        RSI formülünde kullandığı üstel ağırlıklı düzeltme (EMA benzeri,
        alpha = 1/period) kullanılır. Bu yöntem literatürdeki "gerçek" RSI
        hesabıdır ve TradingView/MetaTrader gibi platformlarla uyumludur.
        """
        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Wilder's Smoothing: alpha = 1/period, adjust=False -> özyinelemeli EMA
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # Kayıp ortalaması 0 ise (sürekli yükseliş) RSI 100 olmalı
        rsi = rsi.where(avg_loss != 0, 100.0)

        return rsi

    @staticmethod
    def calculate_sma(series: pd.Series, period: int) -> pd.Series:
        """Basit Hareketli Ortalama (Simple Moving Average)."""
        return series.rolling(window=period, min_periods=period).mean()

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Wilder's Smoothing ile Average True Range (ATR) hesaplar.

        True Range = max(Yüksek-Düşük, |Yüksek-ÖncekiKapanış|,
                          |Düşük-ÖncekiKapanış|)

        ATR, bir hissenin GÜNLÜK ORTALAMA DALGALANMA GENİŞLİĞİNİ TL
        cinsinden ölçer. Sabit yüzdelik bir stop-loss (ör. -%2) her
        hisseye aynı mesafeyi verir; oysa oynak bir hisse için bu çok dar,
        sakin bir hisse için gereksiz geniş olabilir. ATR tabanlı stop,
        stop mesafesini hissenin KENDİ volatilitesine göre ayarlar.
        """
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        return atr

    @staticmethod
    def calculate_rsi_slope(rsi: pd.Series, lookback: int = 3) -> pd.Series:
        """
        RSI'nin son `lookback` gündeki değişimini (eğimini) hesaplar.

        Pozitif değer: RSI yükseliyor, yani momentum güçleniyor. Sadece
        "RSI 50-67 arasında" demek statik bir durumdur; hisse orada günlerce
        hareketsiz kalabilir. Eğim, momentumun GERÇEKTEN artıp artmadığını
        gösterir ve kısa vadeli patlama öncesi ivmelenmeyi yakalamaya
        yardımcı olur.
        """
        return rsi.diff(lookback)

    @staticmethod
    def calculate_bollinger_bandwidth(
        close: pd.Series, period: int = 20, std_multiplier: float = 2.0
    ) -> pd.Series:
        """
        Bollinger Bant Genişliği = (Üst Bant - Alt Bant) / Orta Bant (SMA).

        Düşük bant genişliği, fiyatın dar bir aralıkta sıkıştığını
        (düşük volatilite) gösterir. Klasik teknik analizde "sıkışma"
        (squeeze) döneminin ardından genelde sert ve yönlü bir hareket
        (patlama) gelir. Bu nedenle bant genişliğinin tarihsel olarak
        ne kadar dar olduğunu bilmek, patlama öncesi kurulumu yakalamada
        değerlidir.
        """
        middle = close.rolling(window=period, min_periods=period).mean()
        std = close.rolling(window=period, min_periods=period).std()
        upper = middle + (std * std_multiplier)
        lower = middle - (std * std_multiplier)
        bandwidth = (upper - lower) / middle
        return bandwidth

    @staticmethod
    def calculate_bb_percentile(bandwidth: pd.Series, lookback: int = 100) -> pd.Series:
        """
        Şu anki bant genişliğinin, GEÇMİŞ `lookback` günlük dağılım
        içindeki yüzdelik sırasını hesaplar (0.0 = son `lookback` günün
        EN DAR hali yani en sıkışık; 1.0 = en geniş hali).

        KRİTİK: Hacim ortalamasındaki kendine-referans hatasıyla aynı
        tuzağa düşmemek için, bugünün bant genişliği KENDİ karşılaştırma
        setine dahil edilmez; sadece GEÇMİŞ günlerle kıyaslanır (ileriye
        bakma / kendine referans verme riski yoktur).
        """
        window_size = lookback + 1  # geçmiş `lookback` gün + bugün

        def _percentile_rank(window: np.ndarray) -> float:
            current = window[-1]
            past = window[:-1]
            if len(past) == 0 or np.isnan(current):
                return np.nan
            valid_past = past[~np.isnan(past)]
            if len(valid_past) == 0:
                return np.nan
            return float((valid_past < current).sum()) / len(valid_past)

        return bandwidth.rolling(window=window_size, min_periods=window_size).apply(
            _percentile_rank, raw=True
        )

    @classmethod
    def enrich_dataframe(
        cls,
        df: pd.DataFrame,
        config: ScreeningConfig,
        benchmark_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Ham OHLCV verisine tüm teknik göstergeleri ekleyerek zenginleştirilmiş
        bir dataframe döndürür.

        `benchmark_df` verilirse (ör. XU100 endeks verisi), hissenin
        endekse göre relatif gücü de hesaplanır. Tüm göstergeler SADECE
        GEÇMİŞ VERİYİ kullanan (causal / ileriye bakmayan) yöntemlerle
        hesaplanır; bu sayede aynı fonksiyon hem canlı taramada hem de
        Backtester içinde genişleyen pencerelerle güvenle tekrar
        kullanılabilir.
        """
        df = df.copy()

        # RSI(14) - Wilder's Smoothing
        df["RSI"] = cls.calculate_rsi(df["Close"], period=config.RSI_PERIOD)
        df["RSI_Egim"] = cls.calculate_rsi_slope(df["RSI"], lookback=config.RSI_SLOPE_LOOKBACK)

        # SMA20 / SMA50 (fiyat trendi)
        df["SMA20"] = cls.calculate_sma(df["Close"], config.SMA_SHORT_PERIOD)
        df["SMA50"] = cls.calculate_sma(df["Close"], config.SMA_LONG_PERIOD)

        # Hacim_SMA20 (ortalama işlem hacmi).
        # KRİTİK: Ortalama, BUGÜNÜN hacmini DIŞARIDA bırakacak şekilde bir
        # gün kaydırılarak (shift) hesaplanır. Aksi halde bugünün hacim
        # patlaması kendi ortalamasını da şişirir ve "Son_Hacim /
        # Hacim_SMA20" oranını yapay şekilde küçültüp asıl hacim
        # sıçramalarının filtreden kaçmasına (yanlış negatif) neden olur.
        # Bu yöntem, hacim sıçramasını her zaman "geçmiş 20 günün normal
        # ortalamasına" göre kıyaslar.
        df["Hacim_SMA20"] = cls.calculate_sma(
            df["Volume"].shift(1), config.VOLUME_SMA_PERIOD
        )

        # Çoklu gün hacim teyidi: son N günün (bugün dahil) ortalama hacmi.
        # Tek günlük bir hacim sıçraması (ör. tek bir blok işlem) yanıltıcı
        # olabilir; birkaç günün ortalaması sürdürülebilir bir ilgi artışı
        # olup olmadığını gösterir.
        df["Hacim_Teyit_Ort"] = df["Volume"].rolling(
            window=config.VOLUME_CONFIRM_DAYS, min_periods=config.VOLUME_CONFIRM_DAYS
        ).mean()

        # Günlük TL İşlem Hacmi = Kapanış * Hacim
        df["Gunluk_TL_Hacmi"] = df["Close"] * df["Volume"]

        # ATR(14) - Wilder's Smoothing ve ATR tabanlı dinamik stop
        df["ATR"] = cls.calculate_atr(df["High"], df["Low"], df["Close"], period=config.ATR_PERIOD)
        df["Stop_ATR"] = df["Close"] - (df["ATR"] * config.ATR_STOP_MULTIPLIER)

        # Bollinger Bant Genişliği ve tarihsel yüzdelik sırası (sıkışma tespiti)
        df["BB_Genislik"] = cls.calculate_bollinger_bandwidth(
            df["Close"], period=config.BB_PERIOD, std_multiplier=config.BB_STD_MULTIPLIER
        )
        df["BB_Percentile"] = cls.calculate_bb_percentile(
            df["BB_Genislik"], lookback=config.SQUEEZE_LOOKBACK
        )

        # Relatif Güç: hissenin N günlük getirisi - endeksin N günlük getirisi
        if benchmark_df is not None and not benchmark_df.empty and "Close" in benchmark_df.columns:
            aligned_benchmark_close = benchmark_df["Close"].reindex(df.index).ffill()
            stock_return = df["Close"].pct_change(config.RELATIVE_STRENGTH_PERIOD)
            benchmark_return = aligned_benchmark_close.pct_change(config.RELATIVE_STRENGTH_PERIOD)
            df["Relatif_Guc"] = stock_return - benchmark_return
        else:
            df["Relatif_Guc"] = np.nan

        return df


# ---------------------------------------------------------------------------
# 4. TARAMA STRATEJİSİ (SCREENING STRATEGY)
# ---------------------------------------------------------------------------
@dataclass
class ScanResult:
    """Tek bir hissenin tarama sonucunu tutan basit veri sınıfı."""

    ticker: str
    kapanis: float
    rsi: float
    hacim_karti: float  # Son hacmin, hacim ortalamasına oranı (kat)
    gunluk_tl_hacmi: float
    sma20: float
    sma50: float
    stop_loss: float  # Sabit yüzdelik stop (ör. -%2)
    # --- Gelişmiş göstergeler (bilgi amaçlı; None olabilir) ---
    atr: Optional[float] = None
    stop_loss_atr: Optional[float] = None  # ATR tabanlı dinamik stop
    rsi_egim: Optional[float] = None
    bb_percentile: Optional[float] = None  # 0'a yakın = sıkışma
    relatif_guc: Optional[float] = None  # XU100'e göre göreceli performans
    df: pd.DataFrame = field(default=None, repr=False)


class ScreenerStrategy:
    """
    ScreeningConfig içindeki tüm kriterleri aynı anda sağlayan hisseleri
    seçen tarama stratejisi sınıfı.

    Not: `evaluate_single`, sadece verilen dataframe'in EN SON satırına
    bakar. Bu tasarım bilinçlidir: hem canlı taramada (tüm geçmiş +
    bugün) hem de Backtester içinde (geçmişin bir alt kesiti, "o günkü
    bugün") aynı fonksiyon güvenle tekrar kullanılabilir.
    """

    def __init__(self, config: ScreeningConfig) -> None:
        self.config = config

    def evaluate_single(self, ticker: str, df: pd.DataFrame) -> Optional[ScanResult]:
        """
        Tek bir hissenin en güncel verisini strateji kriterlerine göre
        değerlendirir. Kriterlerin tamamı sağlanıyorsa ScanResult döner,
        aksi halde None döner.
        """
        cfg = self.config

        # Temel göstergeler için NaN kontrolü (bunlar olmadan hiçbir
        # değerlendirme yapılamaz). Gelişmiş göstergeler (BB_Percentile,
        # Relatif_Guc) kasıtlı olarak bu zorunlu listede DEĞİL: onlar
        # "fail-open" mantığıyla çalışır (bkz. aşağıdaki ilgili filtreler).
        zorunlu_kolonlar = ["RSI", "SMA20", "SMA50", "Hacim_SMA20", "ATR"]
        if df[zorunlu_kolonlar].iloc[-1].isna().any():
            return None

        if len(df) < 2:
            return None

        son = df.iloc[-1]
        dun = df.iloc[-2]

        son_kapanis = float(son["Close"])
        son_hacim = float(son["Volume"])
        hacim_sma20 = float(son["Hacim_SMA20"])
        gunluk_tl_hacmi = float(son["Gunluk_TL_Hacmi"])
        rsi = float(son["RSI"])
        sma20 = float(son["SMA20"])
        sma50 = float(son["SMA50"])
        atr = float(son["ATR"])

        dunku_kapanis = float(dun["Close"])
        dunku_en_yuksek = float(dun["High"])

        if hacim_sma20 <= 0 or dunku_en_yuksek <= 0:
            return None

        hacim_karti = son_hacim / hacim_sma20

        # --- Filtre 1: Hacim patlaması ---
        filtre_hacim = son_hacim > (hacim_sma20 * cfg.VOL_MULTIPLIER)

        # --- Filtre 2: Minimum TL likiditesi ---
        filtre_likidite = gunluk_tl_hacmi >= cfg.MIN_TRY_VOLUME

        # --- Filtre 3: RSI bandı ---
        filtre_rsi = cfg.RSI_MIN <= rsi <= cfg.RSI_MAX

        # --- Filtre 4: Trend (fiyat, kısa ve uzun ortalamaların üzerinde) ---
        filtre_trend = (son_kapanis > sma20) and (son_kapanis > sma50)

        # --- Filtre 5: Tavan engeli (dünkü kapanış, dünkü tepeye çok yakın olmasın) ---
        filtre_tavan_engeli = dunku_kapanis < (dunku_en_yuksek * cfg.CEILING_GUARD_RATIO)

        # --- Filtre 6 (opsiyonel): RSI momentum eğimi yükseliyor mu? ---
        rsi_egim_deger = son.get("RSI_Egim", np.nan)
        rsi_egim = None if pd.isna(rsi_egim_deger) else float(rsi_egim_deger)
        if cfg.REQUIRE_RSI_RISING:
            filtre_rsi_egim = (rsi_egim is not None) and (rsi_egim > 0)
        else:
            filtre_rsi_egim = True  # filtre kapalıysa her zaman geçer

        # --- Filtre 7 (opsiyonel): Çoklu gün hacim teyidi ---
        hacim_teyit_deger = son.get("Hacim_Teyit_Ort", np.nan)
        if cfg.REQUIRE_VOLUME_CONFIRMATION:
            filtre_hacim_teyit = (
                not pd.isna(hacim_teyit_deger)
                and hacim_teyit_deger > (hacim_sma20 * cfg.VOLUME_CONFIRM_MULTIPLIER)
            )
        else:
            filtre_hacim_teyit = True

        # --- Filtre 8 (opsiyonel): Volatilite sıkışması (squeeze) ---
        bb_percentile_deger = son.get("BB_Percentile", np.nan)
        bb_percentile = None if pd.isna(bb_percentile_deger) else float(bb_percentile_deger)
        if cfg.REQUIRE_VOLATILITY_SQUEEZE:
            # Gösterge henüz hesaplanamadıysa (yetersiz geçmiş) filtre
            # "fail-open" davranır: hisseyi otomatik elemez, sadece bu
            # ek kriteri değerlendirmeden geçer. Bu sayede backtest'in
            # ilk `SQUEEZE_LOOKBACK` günü tamamen boş dönmez.
            filtre_squeeze = (bb_percentile is None) or (bb_percentile <= cfg.SQUEEZE_PERCENTILE_MAX)
        else:
            filtre_squeeze = True

        # --- Filtre 9 (opsiyonel): XU100'e göre pozitif relatif güç ---
        relatif_guc_deger = son.get("Relatif_Guc", np.nan)
        relatif_guc = None if pd.isna(relatif_guc_deger) else float(relatif_guc_deger)
        if cfg.REQUIRE_POSITIVE_RELATIVE_STRENGTH:
            # Benchmark verisi çekilemediyse (ör. ağ sorunu) bu filtre de
            # fail-open çalışır; yoksa TÜM sonuçlar sessizce kaybolur.
            filtre_relatif_guc = (relatif_guc is None) or (relatif_guc > 0)
        else:
            filtre_relatif_guc = True

        tum_filtreler = (
            filtre_hacim
            and filtre_likidite
            and filtre_rsi
            and filtre_trend
            and filtre_tavan_engeli
            and filtre_rsi_egim
            and filtre_hacim_teyit
            and filtre_squeeze
            and filtre_relatif_guc
        )

        if tum_filtreler:
            stop_loss = son_kapanis * (1 - cfg.STOP_LOSS_PCT)
            stop_loss_atr = son_kapanis - (atr * cfg.ATR_STOP_MULTIPLIER)
            return ScanResult(
                ticker=ticker,
                kapanis=son_kapanis,
                rsi=rsi,
                hacim_karti=hacim_karti,
                gunluk_tl_hacmi=gunluk_tl_hacmi,
                sma20=sma20,
                sma50=sma50,
                stop_loss=stop_loss,
                atr=atr,
                stop_loss_atr=stop_loss_atr,
                rsi_egim=rsi_egim,
                bb_percentile=bb_percentile,
                relatif_guc=relatif_guc,
                df=df,
            )

        return None

    def run(
        self,
        data_map: Dict[str, pd.DataFrame],
        benchmark_df: Optional[pd.DataFrame] = None,
    ) -> List[ScanResult]:
        """Tüm hisse verisi sözlüğü üzerinden taramayı çalıştırır."""
        results: List[ScanResult] = []
        for ticker, raw_df in data_map.items():
            try:
                enriched = TechnicalAnalyzer.enrich_dataframe(raw_df, self.config, benchmark_df=benchmark_df)
                res = self.evaluate_single(ticker, enriched)
                if res is not None:
                    results.append(res)
            except Exception as exc:
                logger.error(f"{ticker}: Strateji değerlendirmesi sırasında hata -> {exc}")
                continue
        return results


# ---------------------------------------------------------------------------
# 5. BACKTESTER (GERİYE DÖNÜK TEST MOTORU)
# ---------------------------------------------------------------------------
class Backtester:
    """
    Stratejinin GEÇMİŞTE ne zaman sinyal ürettiğini bulup, her sinyalden
    sonraki N günlük fiyat performansını hesaplayan geriye dönük test
    motoru.

    Neden önemli: "RSI 50-67 ve hacim 1.8 katı" gibi kurallar sezgiyle
    kurulmuştur ve hiç doğrulanmadan kullanılırsa gerçek bir tahmin gücü
    olup olmadığı bilinmez. Backtester, ScreenerStrategy.evaluate_single
    fonksiyonunu genişleyen (expanding) pencerelerle geçmişteki her güne
    uygulayarak "bu kurallar geçmişte kaç kez tetiklendi ve ortalama
    getirisi ne oldu" sorusuna somut bir cevap verir.
    """

    def __init__(self, config: ScreeningConfig) -> None:
        self.config = config
        self.strategy = ScreenerStrategy(config)

    def run_single(
        self,
        ticker: str,
        enriched_df: pd.DataFrame,
        forward_days: int,
    ) -> pd.DataFrame:
        """
        Bir hissenin tüm geçmişini gün gün gezip, her günde strateji
        sinyali tetiklenmiş mi diye bakar; tetiklenmişse `forward_days`
        gün sonraki getiriyi hesaplar.

        Not: `enriched_df.iloc[: i + 1]` ile SADECE o güne kadar olan
        veri stratejiye verilir; bu, ileriye bakma (look-ahead) hatasını
        önler -- göstergeler zaten sadece geçmişe bakan (rolling/ewm)
        yöntemlerle hesaplandığından, dilimleme ekstra bir güvenlik katmanıdır.
        """
        records = []
        n = len(enriched_df)
        min_start = max(self.config.MIN_HISTORY_DAYS, self.config.SQUEEZE_LOOKBACK + 1)

        for i in range(min_start, n - forward_days):
            window_df = enriched_df.iloc[: i + 1]
            try:
                result = self.strategy.evaluate_single(ticker, window_df)
            except Exception as exc:
                logger.error(f"{ticker}: Backtest degerlendirmesinde hata -> {exc}")
                continue

            if result is not None:
                entry_price = result.kapanis
                exit_price = float(enriched_df["Close"].iloc[i + forward_days])
                forward_return_pct = (exit_price / entry_price - 1.0) * 100.0
                records.append(
                    {
                        "Tarih": enriched_df.index[i],
                        "Hisse": ticker.replace(".IS", ""),
                        "Giris_Fiyati": round(entry_price, 2),
                        "Cikis_Fiyati": round(exit_price, 2),
                        "Getiri_%": round(forward_return_pct, 2),
                    }
                )

        return pd.DataFrame(records)

    def run_universe(
        self,
        data_map: Dict[str, pd.DataFrame],
        forward_days: int,
        benchmark_df: Optional[pd.DataFrame] = None,
        max_tickers: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Tüm hisse evreni (veya bir alt kümesi) için backtest çalıştırıp
        birleşik sonuç dataframe'i döndürür.

        `max_tickers`: Streamlit arayüzünde her seferinde tüm BIST 100'ü
        geriye dönük taramak yavaş olabileceğinden, kullanıcı isterse
        bunu sınırlı bir alt kümeyle (ör. son taramada sinyal veren
        hisselerle) sınırlayabilir.
        """
        all_records: List[pd.DataFrame] = []
        items = list(data_map.items())
        if max_tickers is not None:
            items = items[:max_tickers]

        for ticker, raw_df in items:
            try:
                enriched = TechnicalAnalyzer.enrich_dataframe(raw_df, self.config, benchmark_df=benchmark_df)
                res_df = self.run_single(ticker, enriched, forward_days)
                if not res_df.empty:
                    all_records.append(res_df)
            except Exception as exc:
                logger.error(f"{ticker}: Backtest calisirken hata -> {exc}")
                continue

        if not all_records:
            return pd.DataFrame(columns=["Tarih", "Hisse", "Giris_Fiyati", "Cikis_Fiyati", "Getiri_%"])

        return pd.concat(all_records, ignore_index=True)

    @staticmethod
    def summarize(backtest_df: pd.DataFrame) -> Dict[str, float]:
        """Backtest sonuçlarından özet istatistik (isabet oranı, ort. getiri vb.) çıkarır."""
        if backtest_df.empty or "Getiri_%" not in backtest_df.columns:
            return {
                "sinyal_sayisi": 0,
                "isabet_orani": 0.0,
                "ortalama_getiri": 0.0,
                "medyan_getiri": 0.0,
            }
        returns = backtest_df["Getiri_%"]
        return {
            "sinyal_sayisi": int(len(returns)),
            "isabet_orani": float((returns > 0).mean() * 100),
            "ortalama_getiri": float(returns.mean()),
            "medyan_getiri": float(returns.median()),
        }


# ---------------------------------------------------------------------------
# 6. SİNYAL GEÇMİŞİ DEPOSU (SIGNAL HISTORY STORE)
# ---------------------------------------------------------------------------
class SignalHistoryStore:
    """
    Her taramada üretilen sinyalleri bir CSV dosyasına kaydeden ve daha
    sonra -- yeterli gün geçtiğinde -- bu sinyallerin GERÇEKTE ne kadar
    getiri sağladığını (ileri tarihli fiyat verisiyle) geriye dönük
    dolduran basit, kalıcı bir sinyal günlüğü.

    Neden önemli: Backtester geçmiş veriyle "sanal" bir test yaparken,
    bu sınıf GERÇEK ZAMANLI olarak üretilen sinyalleri kayıt altına
    alır. Böylece zaman içinde "gerçekten kaç sinyal isabetli çıktı"
    sorusuna, canlı kullanım verisiyle cevap verilebilir.
    """

    HISTORY_COLUMNS = [
        "Tarih", "Hisse", "Kapanis", "RSI", "Hacim_Karti",
        f"Sonraki_Getiri_%",
    ]

    def __init__(self, file_path: str = "signal_history.csv") -> None:
        self.file_path = file_path

    def append_signals(self, results: List[ScanResult], scan_date: Optional[pd.Timestamp] = None) -> None:
        """Bugünkü tarama sonuçlarını geçmiş dosyasına ekler (varsa üzerine yazmaz, tekrarı önler)."""
        if not results:
            return

        scan_date = scan_date or pd.Timestamp(datetime.now().date())
        new_rows = [
            {
                "Tarih": scan_date.strftime("%Y-%m-%d"),
                "Hisse": r.ticker,
                "Kapanis": r.kapanis,
                "RSI": round(r.rsi, 1),
                "Hacim_Karti": round(r.hacim_karti, 2),
                "Sonraki_Getiri_%": np.nan,
            }
            for r in results
        ]
        new_df = pd.DataFrame(new_rows)

        try:
            if os.path.exists(self.file_path):
                existing = pd.read_csv(self.file_path)
                combined = pd.concat([existing, new_df], ignore_index=True)
                # Aynı gün + aynı hisse tekrar kaydedilirse (ör. kullanıcı
                # aynı gün birden fazla tarama yaparsa) yalnızca en son
                # kaydı tut.
                combined = combined.drop_duplicates(subset=["Tarih", "Hisse"], keep="last")
            else:
                combined = new_df
            combined.to_csv(self.file_path, index=False)
        except Exception as exc:
            logger.error(f"Sinyal gecmisi kaydedilirken hata: {exc}")

    def load_history(self) -> pd.DataFrame:
        """Kayıtlı sinyal geçmişini okur; dosya yoksa boş bir dataframe döner."""
        try:
            if os.path.exists(self.file_path):
                return pd.read_csv(self.file_path)
        except Exception as exc:
            logger.error(f"Sinyal gecmisi okunurken hata: {exc}")
        return pd.DataFrame(columns=self.HISTORY_COLUMNS)

    def update_outcomes(self, data_map: Dict[str, pd.DataFrame], forward_days: int) -> pd.DataFrame:
        """
        Daha önce kaydedilmiş ama henüz sonucu (ileri getirisi) bilinmeyen
        sinyaller için, eğer o hissenin güncel verisinde sinyal tarihinden
        itibaren yeterli gün geçmişse, gerçekleşen getiriyi hesaplayıp
        doldurur ve dosyayı günceller.
        """
        history = self.load_history()
        if history.empty:
            return history

        if "Sonraki_Getiri_%" not in history.columns:
            history["Sonraki_Getiri_%"] = np.nan

        degisiklik_oldu = False

        for idx, row in history.iterrows():
            if pd.notna(row.get("Sonraki_Getiri_%")):
                continue  # zaten hesaplanmış

            ticker = row["Hisse"]
            if ticker not in data_map:
                continue

            df = data_map[ticker]
            try:
                signal_date = pd.Timestamp(row["Tarih"])
            except Exception:
                continue

            future_dates = df.index[df.index.normalize() > signal_date.normalize()]
            if len(future_dates) < forward_days:
                continue  # henüz yeterli gün geçmemiş

            exit_date = future_dates[forward_days - 1]
            try:
                entry_price = float(row["Kapanis"])
                exit_price = float(df.loc[exit_date, "Close"])
                if entry_price > 0:
                    history.at[idx, "Sonraki_Getiri_%"] = round((exit_price / entry_price - 1.0) * 100, 2)
                    degisiklik_oldu = True
            except Exception as exc:
                logger.error(f"{ticker}: gecmis getiri hesaplanirken hata -> {exc}")

        if degisiklik_oldu:
            try:
                history.to_csv(self.file_path, index=False)
            except Exception as exc:
                logger.error(f"Sinyal gecmisi guncellenirken hata: {exc}")

        return history


# ---------------------------------------------------------------------------
# 7. GRAFİK OLUŞTURUCU (PLOTLY - Çift Panelli Mum + Hacim Grafiği)
# ---------------------------------------------------------------------------
class ChartBuilder:
    """Plotly ile interaktif mum grafiği + hacim panelini üreten yardımcı sınıf."""

    @staticmethod
    def build_candlestick_volume_chart(
        df: pd.DataFrame, ticker: str, config: ScreeningConfig
    ) -> go.Figure:
        """
        Üstte fiyat mumu (candlestick), altta işlem hacmi olacak şekilde
        çift panelli, karanlık temalı interaktif grafik oluşturur.
        """
        plot_df = df.tail(config.CHART_LOOKBACK_DAYS).copy()

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.72, 0.28],
            subplot_titles=(f"{ticker} - Fiyat", "İşlem Hacmi"),
        )

        # --- Üst panel: Mum grafiği ---
        fig.add_trace(
            go.Candlestick(
                x=plot_df.index,
                open=plot_df["Open"],
                high=plot_df["High"],
                low=plot_df["Low"],
                close=plot_df["Close"],
                name="Fiyat",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
            ),
            row=1,
            col=1,
        )

        # SMA çizgilerini üst panele ekle
        if "SMA20" in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index, y=plot_df["SMA20"],
                    mode="lines", name="SMA20",
                    line=dict(color="#42a5f5", width=1.3),
                ),
                row=1, col=1,
            )
        if "SMA50" in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index, y=plot_df["SMA50"],
                    mode="lines", name="SMA50",
                    line=dict(color="#ffca28", width=1.3),
                ),
                row=1, col=1,
            )

        # --- Alt panel: Hacim çubukları (yükselişte yeşil, düşüşte kırmızı) ---
        renk = np.where(plot_df["Close"] >= plot_df["Open"], "#26a69a", "#ef5350")
        fig.add_trace(
            go.Bar(
                x=plot_df.index,
                y=plot_df["Volume"],
                name="Hacim",
                marker_color=renk,
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            template="plotly_dark",
            height=650,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            margin=dict(l=40, r=40, t=60, b=40),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_yaxes(title_text="Fiyat (TL)", row=1, col=1)
        fig.update_yaxes(title_text="Hacim", row=2, col=1)

        return fig

    @staticmethod
    def build_return_histogram(backtest_df: pd.DataFrame, forward_days: int) -> go.Figure:
        """Backtest'teki ileri getirilerin dağılımını histogram olarak çizer."""
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=backtest_df["Getiri_%"],
                nbinsx=30,
                marker_color="#42a5f5",
                name="Getiri Dağılımı",
            )
        )
        fig.add_vline(x=0, line_dash="dash", line_color="#ef5350")
        fig.update_layout(
            template="plotly_dark",
            height=350,
            title=f"{forward_days} Günlük İleri Getiri Dağılımı",
            xaxis_title="Getiri (%)",
            yaxis_title="Sinyal Sayısı",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            margin=dict(l=40, r=40, t=50, b=40),
        )
        return fig


# ---------------------------------------------------------------------------
# 8. STREAMLIT DASHBOARD UYGULAMASI (ANA UYGULAMA SINIFI)
# ---------------------------------------------------------------------------
class DashboardApp:
    """
    Tüm Streamlit arayüz mantığını, durum yönetimini ve kullanıcı
    etkileşimlerini koordine eden ana uygulama sınıfı.
    """

    def __init__(self) -> None:
        self.data_provider = BistDataProvider()
        self._init_session_state()

    # ------------------------- Durum Yönetimi -------------------------
    @staticmethod
    def _init_session_state() -> None:
        """
        Kullanıcı hisse seçtiğinde ya da sayfa yeniden çizildiğinde
        tarama sonuçlarının kaybolmaması için gerekli session_state
        anahtarlarını başlatır.
        """
        defaults = {
            "scan_results": [],
            "data_map": {},
            "benchmark_df": None,
            "total_scanned": 0,
            "scan_completed": False,
            "backtest_df": None,
            "backtest_forward_days": None,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    # ------------------------- Sayfa Yapılandırması -------------------------
    @staticmethod
    def configure_page() -> None:
        """Sayfa genel ayarlarını ve karanlık tema odaklı CSS'i uygular."""
        st.set_page_config(
            page_title="BIST 100 Hisse Tarayıcı",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Modern, koyu temalı finans paneli görünümü için özel CSS
        st.markdown(
            """
            <style>
                .stApp { background-color: #0e1117; }
                [data-testid="stMetricValue"] { font-size: 1.8rem; color: #26a69a; }
                [data-testid="stMetricLabel"] { color: #b0b0b0; }
                div[data-testid="stMetric"] {
                    background-color: #161b22;
                    border: 1px solid #262c36;
                    border-radius: 10px;
                    padding: 12px 16px;
                }
                h1, h2, h3 { color: #e6edf3; }
                .stDataFrame { border-radius: 8px; overflow: hidden; }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # ------------------------- Sidebar (Ayarlar Paneli) -------------------------
    def render_sidebar(self) -> Tuple[ScreeningConfig, bool]:
        """
        Sol panelde strateji parametrelerinin ayarlanabildiği slider'ları,
        gelişmiş (opsiyonel) filtreleri ve 'Taramayı Başlat' butonunu
        oluşturur. Güncel ScreeningConfig nesnesini ve buton durumunu
        döndürür.
        """
        with st.sidebar:
            st.header("⚙️ Tarama Ayarları")
            st.caption("Strateji parametrelerini ihtiyacınıza göre ayarlayın.")

            st.subheader("RSI Bandı")
            rsi_min, rsi_max = st.slider(
                "RSI Aralığı (Min - Max)",
                min_value=0, max_value=100,
                value=(50, 67), step=1,
                help="Sadece bu RSI aralığındaki hisseler taranır.",
            )

            st.subheader("Hacim Kriterleri")
            vol_multiplier = st.slider(
                "Hacim Çarpanı (Son Hacim / Ort. Hacim)",
                min_value=1.0, max_value=5.0, value=1.8, step=0.1,
            )
            min_try_volume = st.slider(
                "Minimum Günlük TL Hacmi (Milyon TL)",
                min_value=1, max_value=500, value=50, step=1,
            ) * 1_000_000

            st.subheader("Risk Yönetimi")
            stop_loss_pct = st.slider(
                "Stop Loss Yüzdesi (%)",
                min_value=0.5, max_value=10.0, value=2.0, step=0.5,
            ) / 100
            atr_stop_multiplier = st.slider(
                "ATR Stop Çarpanı (dinamik stop)",
                min_value=0.5, max_value=4.0, value=1.5, step=0.1,
                help="Dinamik stop = Kapanış - (ATR × bu çarpan). Oynak hisselerde otomatik olarak daha geniş bir stop verir.",
            )

            # ------------------- Gelişmiş Filtreler (opsiyonel) -------------------
            with st.expander("🧪 Gelişmiş Filtreler (opsiyonel)", expanded=False):
                st.caption(
                    "Bu filtreler varsayılan olarak KAPALIDIR. Açmadan önce "
                    "aşağıdaki Backtest bölümüyle gerçekten işe yarayıp "
                    "yaramadığını ölçmenizi öneririz."
                )
                require_rsi_rising = st.checkbox(
                    "RSI momentumu yükseliyor olsun (RSI eğimi > 0)",
                    value=False,
                )
                require_volume_confirm = st.checkbox(
                    "Çoklu gün hacim teyidi (tek günlük hacim gürültüsünü azalt)",
                    value=False,
                )
                require_squeeze = st.checkbox(
                    "Volatilite sıkışması sonrası olsun (Bollinger squeeze)",
                    value=False,
                )
                require_relative_strength = st.checkbox(
                    "XU100'e göre pozitif relatif güç şart olsun",
                    value=False,
                )

            st.divider()
            start_button = st.button("🚀 Taramayı Başlat", use_container_width=True, type="primary")

            st.divider()
            st.caption(f"Tarama evreni: BIST 100 ({len(self.data_provider.tickers)} hisse)")
            st.caption("Veri kaynağı: Yahoo Finance (yfinance) — 1 saat önbellek")

        config = ScreeningConfig(
            RSI_MIN=float(rsi_min),
            RSI_MAX=float(rsi_max),
            VOL_MULTIPLIER=float(vol_multiplier),
            MIN_TRY_VOLUME=float(min_try_volume),
            STOP_LOSS_PCT=float(stop_loss_pct),
            ATR_STOP_MULTIPLIER=float(atr_stop_multiplier),
            REQUIRE_RSI_RISING=bool(require_rsi_rising),
            REQUIRE_VOLUME_CONFIRMATION=bool(require_volume_confirm),
            REQUIRE_VOLATILITY_SQUEEZE=bool(require_squeeze),
            REQUIRE_POSITIVE_RELATIVE_STRENGTH=bool(require_relative_strength),
        )
        return config, start_button

    # ------------------------- Tarama İşlemi -------------------------
    def run_scan(self, config: ScreeningConfig) -> None:
        """Veri çekimini ve taramayı çalıştırıp sonuçları session_state'e kaydeder."""
        with st.spinner("BIST 100 hisseleri taranıyor, lütfen bekleyin..."):
            tickers_tuple: Tuple[str, ...] = tuple(self.data_provider.tickers)

            data_map = BistDataProvider.fetch_data(
                tickers_tuple,
                period=config.DOWNLOAD_PERIOD,
                interval=config.DOWNLOAD_INTERVAL,
                min_history_days=config.MIN_HISTORY_DAYS,
            )

            # XU100 endeks verisini (relatif güç için) ayrıca çek. Ağ
            # sorunu ya da sembol bulunamaması durumunda strateji
            # "fail-open" çalışacağı için uygulama çökmez.
            benchmark_map = BistDataProvider.fetch_data(
                (config.BENCHMARK_TICKER,),
                period=config.DOWNLOAD_PERIOD,
                interval=config.DOWNLOAD_INTERVAL,
                min_history_days=config.RELATIVE_STRENGTH_PERIOD + 5,
            )
            benchmark_df = benchmark_map.get(config.BENCHMARK_TICKER)
            if benchmark_df is None:
                logger.warning(
                    f"Benchmark ({config.BENCHMARK_TICKER}) verisi alınamadı; "
                    "relatif güç göstergesi bu taramada boş kalacak."
                )

            strategy = ScreenerStrategy(config)
            results = strategy.run(data_map, benchmark_df=benchmark_df)

            # Sonuçları hacim katına göre azalan sırada sırala
            results.sort(key=lambda r: r.hacim_karti, reverse=True)

            st.session_state.scan_results = results
            st.session_state.data_map = data_map
            st.session_state.benchmark_df = benchmark_df
            st.session_state.total_scanned = len(data_map)
            st.session_state.scan_completed = True

            # Sinyal geçmişini güncelle: bugünkü sinyalleri kaydet, daha
            # önce kaydedilmiş ve artık ileri getirisi hesaplanabilecek
            # sinyalleri de doldur.
            history_store = SignalHistoryStore(config.SIGNAL_HISTORY_FILE)
            history_store.append_signals(results)
            history_store.update_outcomes(data_map, config.BACKTEST_FORWARD_DAYS)

    # ------------------------- Metrik Kartları -------------------------
    @staticmethod
    def render_metrics(results: List[ScanResult], total_scanned: int) -> None:
        """Ekranın üstüne 3 adet özet metrik kartı yerleştirir."""
        col1, col2, col3 = st.columns(3)

        max_hacim_karti = max((r.hacim_karti for r in results), default=0.0)

        with col1:
            st.metric("Toplam Taranan Hisse", f"{total_scanned}")
        with col2:
            st.metric("Sinyal Veren Hisse Sayısı", f"{len(results)}")
        with col3:
            st.metric("Günün En Yüksek Hacim Katı", f"{max_hacim_karti:.2f}x")

    # ------------------------- Sonuç Tablosu -------------------------
    @staticmethod
    def render_results_table(results: List[ScanResult]) -> pd.DataFrame:
        """
        Tarama sonuçlarını renklendirilmiş bir tabloda gösterir.
        Görüntülenen dataframe'i döndürür (selectbox için kullanılacak).
        """
        table_data = []
        for r in results:
            table_data.append(
                {
                    "Hisse": r.ticker.replace(".IS", ""),
                    "Kapanış (TL)": round(r.kapanis, 2),
                    "RSI": round(r.rsi, 1),
                    "RSI Eğimi": round(r.rsi_egim, 2) if r.rsi_egim is not None else None,
                    "Hacim Katı": round(r.hacim_karti, 2),
                    "Günlük TL Hacmi": round(r.gunluk_tl_hacmi, 0),
                    "SMA20": round(r.sma20, 2),
                    "SMA50": round(r.sma50, 2),
                    "Stop Loss (-%)": round(r.stop_loss, 2),
                    "ATR Stop (dinamik)": round(r.stop_loss_atr, 2) if r.stop_loss_atr is not None else None,
                    "Volatilite Percentile": (
                        round(r.bb_percentile * 100, 1) if r.bb_percentile is not None else None
                    ),
                    "Relatif Güç (%)": (
                        round(r.relatif_guc * 100, 2) if r.relatif_guc is not None else None
                    ),
                }
            )

        df_display = pd.DataFrame(table_data)

        format_map = {
            "Kapanış (TL)": "{:.2f}",
            "RSI": "{:.1f}",
            "RSI Eğimi": "{:.2f}",
            "Hacim Katı": "{:.2f}x",
            "Günlük TL Hacmi": "{:,.0f}",
            "SMA20": "{:.2f}",
            "SMA50": "{:.2f}",
            "Stop Loss (-%)": "{:.2f}",
            "ATR Stop (dinamik)": "{:.2f}",
            "Volatilite Percentile": "{:.1f}%",
            "Relatif Güç (%)": "{:+.2f}%",
        }

        try:
            # "Hacim Katı" sütununa yeşil gradyan, "Stop Loss" sütununa
            # kırmızı vurgu uygula. background_gradient dahili olarak
            # matplotlib gerektirir; kurulu değilse ImportError fırlatır.
            styled = (
                df_display.style
                .background_gradient(subset=["Hacim Katı"], cmap="Greens")
                .map(lambda _: "color: #ef5350; font-weight: 600;", subset=["Stop Loss (-%)", "ATR Stop (dinamik)"])
                .format(format_map, na_rep="—")
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)
        except ImportError:
            # matplotlib kurulu değilse gradyansız, düz ama biçimlendirilmiş
            # tabloya güvenli şekilde geri dön (uygulama çökmez).
            logger.warning("matplotlib bulunamadı; tablo gradyansız gösteriliyor.")
            st.caption("ℹ️ Renkli gradyan için `pip install matplotlib` gerekir.")
            styled = df_display.style.format(format_map, na_rep="—")
            st.dataframe(styled, use_container_width=True, hide_index=True)

        return df_display

    # ------------------------- Detay Grafiği -------------------------
    def render_stock_detail_chart(self, results: List[ScanResult], config: ScreeningConfig) -> None:
        """Seçilen hissenin son N günlük mum + hacim grafiğini çizer."""
        st.subheader("📊 Hisse Detay Grafiği")

        ticker_options = [r.ticker for r in results]
        display_options = [t.replace(".IS", "") for t in ticker_options]

        selected_display = st.selectbox("İncelemek istediğiniz hisseyi seçin:", display_options)
        selected_ticker = ticker_options[display_options.index(selected_display)]

        selected_result = next((r for r in results if r.ticker == selected_ticker), None)
        if selected_result is None:
            st.warning("Seçilen hisseye ait veri bulunamadı.")
            return

        fig = ChartBuilder.build_candlestick_volume_chart(selected_result.df, selected_display, config)
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------- Backtest Bölümü -------------------------
    def render_backtest_section(self, config: ScreeningConfig) -> None:
        """
        Kullanıcının mevcut strateji parametreleriyle geçmişte kaç sinyal
        üretilmiş olacağını ve bu sinyallerin ortalama getirisini gösteren
        interaktif backtest bölümü.
        """
        st.subheader("🔬 Backtest — Geçmiş Performans")
        st.caption(
            "Şu anki strateji ayarlarınızla GEÇMİŞTE üretilmiş olacak sinyalleri "
            "bulur ve her birinin belirlediğiniz gün sonraki getirisini hesaplar. "
            "Bu, stratejinizi canlıda kullanmadan önce sağlamasını yapmanın en "
            "güvenilir yoludur."
        )

        if not st.session_state.data_map:
            st.info("Backtest çalıştırmak için önce en az bir kez tarama yapmalısınız.")
            return

        col1, col2 = st.columns([2, 1])
        with col1:
            forward_days = st.slider(
                "İleri getiri periyodu (gün)",
                min_value=1, max_value=20,
                value=config.BACKTEST_FORWARD_DAYS, step=1,
                key="backtest_forward_days_slider",
            )
        with col2:
            st.write("")
            st.write("")
            run_backtest = st.button("▶️ Backtest Çalıştır", use_container_width=True)

        if run_backtest:
            with st.spinner("Geçmiş veriler üzerinde strateji test ediliyor... (bu biraz sürebilir)"):
                backtester = Backtester(config)
                backtest_df = backtester.run_universe(
                    st.session_state.data_map,
                    forward_days=forward_days,
                    benchmark_df=st.session_state.benchmark_df,
                )
                st.session_state.backtest_df = backtest_df
                st.session_state.backtest_forward_days = forward_days

        backtest_df = st.session_state.get("backtest_df")
        if backtest_df is None:
            st.info("Sonuçları görmek için 'Backtest Çalıştır' butonuna basın.")
            return

        if backtest_df.empty:
            st.info(
                "Bu ayarlarla geçmişte hiç sinyal üretilmemiş. Kriterler çok sıkı "
                "olabilir; sol panelden ayarları gevşetip tekrar deneyebilirsiniz."
            )
            return

        used_forward_days = st.session_state.get("backtest_forward_days", forward_days)
        summary = Backtester.summarize(backtest_df)

        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
        with b_col1:
            st.metric("Geçmiş Sinyal Sayısı", f"{summary['sinyal_sayisi']}")
        with b_col2:
            st.metric("İsabet Oranı", f"{summary['isabet_orani']:.1f}%")
        with b_col3:
            st.metric("Ortalama Getiri", f"{summary['ortalama_getiri']:+.2f}%")
        with b_col4:
            st.metric("Medyan Getiri", f"{summary['medyan_getiri']:+.2f}%")

        fig = ChartBuilder.build_return_histogram(backtest_df, used_forward_days)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Tüm geçmiş sinyalleri gör"):
            st.dataframe(
                backtest_df.sort_values("Tarih", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

    # ------------------------- Sinyal Geçmişi Bölümü -------------------------
    @staticmethod
    def render_signal_history_section(config: ScreeningConfig) -> None:
        """Canlı taramalardan biriken gerçek sinyal geçmişini ve gerçekleşen getirileri gösterir."""
        st.subheader("🗂️ Sinyal Geçmişi (Canlı Kayıt)")
        st.caption(
            "Her taramada üretilen sinyaller otomatik olarak "
            f"`{config.SIGNAL_HISTORY_FILE}` dosyasına kaydedilir. Yeterli gün "
            "geçtikçe gerçekleşen getiri otomatik olarak hesaplanıp doldurulur."
        )

        history_store = SignalHistoryStore(config.SIGNAL_HISTORY_FILE)
        history = history_store.load_history()

        if history.empty:
            st.info("Henüz kaydedilmiş bir sinyal geçmişi yok. En az bir tarama yapmanız yeterli.")
            return

        completed = history.dropna(subset=["Sonraki_Getiri_%"]) if "Sonraki_Getiri_%" in history.columns else pd.DataFrame()

        if not completed.empty:
            h_col1, h_col2, h_col3 = st.columns(3)
            with h_col1:
                st.metric("Sonucu Belli Olan Sinyal", f"{len(completed)}")
            with h_col2:
                isabet = (completed["Sonraki_Getiri_%"] > 0).mean() * 100
                st.metric("Gerçek İsabet Oranı", f"{isabet:.1f}%")
            with h_col3:
                st.metric("Gerçek Ortalama Getiri", f"{completed['Sonraki_Getiri_%'].mean():+.2f}%")
        else:
            st.info("Kayıtlı sinyaller var ama henüz hiçbirinin sonucu belirlenecek kadar gün geçmemiş.")

        st.dataframe(
            history.sort_values("Tarih", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    # ------------------------- Ana Çalıştırma Döngüsü -------------------------
    def run(self) -> None:
        """Uygulamanın ana giriş noktası; tüm arayüz akışını yönetir."""
        self.configure_page()

        st.title("📈 BIST 100 Hisse Tarama ve Analiz Dashboard'u")
        st.caption(
            "RSI, hacim patlaması ve trend filtrelerine dayalı, kural tabanlı "
            "teknik tarama aracı. Yatırım tavsiyesi değildir."
        )

        config, start_button = self.render_sidebar()

        if start_button:
            self.run_scan(config)

        st.divider()

        if not st.session_state.scan_completed:
            st.info("👈 Sol panelden ayarları yapılandırıp 'Taramayı Başlat' butonuna basın.")
            return

        results: List[ScanResult] = st.session_state.scan_results
        total_scanned: int = st.session_state.total_scanned

        self.render_metrics(results, total_scanned)
        st.divider()

        if not results:
            st.info("Kriterlere uyan hisse bulunamadı.")
        else:
            st.subheader("🎯 Sinyal Veren Hisseler")
            self.render_results_table(results)
            st.divider()
            self.render_stock_detail_chart(results, config)

        st.divider()
        self.render_backtest_section(config)

        st.divider()
        self.render_signal_history_section(config)

        st.divider()
        st.caption(
            f"Son tarama: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')} | "
            "Veriler Yahoo Finance'ten alınmıştır ve yatırım tavsiyesi niteliği taşımaz."
        )


# ---------------------------------------------------------------------------
# UYGULAMA GİRİŞ NOKTASI
# ---------------------------------------------------------------------------
def main() -> None:
    """Streamlit uygulamasının ana giriş fonksiyonu."""
    app = DashboardApp()
    app.run()


if __name__ == "__main__":
    main()
