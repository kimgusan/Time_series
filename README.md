# ⏰ 시계열 데이터

-   시간이 지남에 따라 기록된 정보들을 의미하고, 관측시간에 대한 관측자료의 관계로 표현한다.
-   시간 흐름에 따라 변화하는 방향성을 이해하고, 그 변화를 예측하는 것이 주 목적이다.
-   여름 휴가 기간 동안 특정 해변 도시의 방문자 수를 기록하고 분석하면, 일정한 패턴이나 추세를 찾게 된다.  
    이를 통해 다음 해의 방문자 수를 예측하여 관광 사업이나 여행 업계에서 효율적인 운영 및 마케팅 전략을 계획할 수 있다.
-   시계열 데이터의 유형에는 우연변동 시계열, 계절 변동 시계열, 추세 변동 시계열, 계절-추세 변동 시계열 데이터가 있다.
-   자료의 인덱스가 시간인 데이터이다.

---

## 목차

1. [시계열 데이터란?](#이론)
    - [정상성이란?](#정상성)
    - 라그, 차분, 자기상관함수(ACF), 부분자기상관함수(PACF)
2. [ARIMA](#ARIMA)
3. [Prophet (비트코인 시장 예측)](#Prophet)

---

## 종류

1. 우연변동 시계열

    - 특정한 패턴이나 추세 없이 우연한 요소에 의해 변하는 데이터이다.
    - 패턴이나 규칙이 없고 우연히 발생하는 것처럼 보인다.

2. 계절변동 시계열 데이터

    - 특정한 시간 주기(예: 일년)에 따라 발생하는 패턴이 나타나는 데이터이다.
    - 특정 시간에 특정한 일이 반복된다.

3. 추세변동 시계열 데이터

    - 시간에 따라 일정한 방향으로 계속해서 증가하거나 감소하는 추세를 보이는 데이터이다.
    - 일정한 방향으로 계속해서 증가하거나 감소한다.

4. 계절-추세변동 시계열 데이터
    - 추세와 계절성 요인이 함께 나타나는 데이터로서, 시간에 따라 일정한 추세가 있고 특정 시간 주기에 따른 패턴이 있는 데이터이다.
    - 시간이 지남에 따라 일정한 방향으로 움직이면서도 특정 시간에 특정한 일이 반복된다.

## <div id='정상성'>정상성 (正常性, Stationarity)</div>

-   일정해서 늘 한결같은 성질을 뜻한다.
-   관측된 시간에 무관하게 과거, 현재, 미래의 분포가 같아야 한다.
-   평균, 분산 등이 변하지 않으며 추세나 계절성이 없는 시계열 데이터이다.
-   하지만, **정상성을 나타내는 시계열은 장기적으로 볼 때 예측할 수 있는 패턴을 나타내지 않아야 한다. 즉, 불규칙 해야한다.**
-   즉, 어떤 특정한 주기로 **반복하는 계절성이나 위로, 아래로 가능 추세성이 없어야 한다.**

## 라그(Lag)

-   라그(시차)는 현재 시점에서 이전 시점의 값을 의미하며, 특정 시점 t에서의 라그는 t-k에서의 값을 가리킨다.
-   시계열 데이터에서 패턴과 트렌드를 분석하고 예측하는 데 중요한 개념이다.
-   예를 들어, 하루 전의 주식 가격을 이용하여 다음 날의 주식 가격을 예측하는 등의 분석에 사용될 수 있다.

## 차분 (Differencing)

-   연이은 관측값들의 차이를 계산해준다.
-   시계열 데이터의 평균과 분산이 일정해야 시계열 값을 예측할 수 있다.
-   정상성을 나타내지 않은 시계열에서 정상성을 나타내도록 만드는 방법 중 대표적인 방법이다.
-   차분을 통해 추세나 계절성을 제거하거나 감소시시킬 수 있다.
-   라그를 사용하여 시계열 데이터를 분석할 때, 라그된 데이터 사이의 차이를 계산하여 차분을 수행한다.

## 자기상관 함수 (Autocorrelation Function, ACF)

-   자기상관이란, 현재 시점에서 이전 시점 간의 관련성을 뜻한다.
-   **시간 흐름에 따라 각 데이터는 독립적이지 않다.**  
     전일 데이터가 금일 데이터에 영향을 주고, 익일 데이터는 금일 데이터의 영향을 받는다.
-   시계열의 라그 사이의 선형 관계를 측정해서 **시계열 자료의 정상성을 파악할 때 사용**한다.
-   ACF 그래프는 정상 시계열(정상성을 띄는 시계열)일 경우 모든 시차에서 0에 근접한 모양을 나타내고,
-   비정상 시계열은 천천히 감소할 경우 추세, 물결 모양일 경우 계절이다.

            1. 시차에 대한 항들이 누적된다.
            2. 현재 시점으로부터 정확히 이전 lag와의 상관관계를 측정하는 것이다.
            3. 시계열 데이터의 전반적인 패턴을 파악해서 추세나 주기성 등 다양한 특성을 확인할 수 있다.
            4. 차분을 통해 정상 시계열로 변환한 뒤 ACF를 구하면, 정상성을 가진 시계열에서의 자기상관을 파악할 수 있다.
            5. 온라인 판매 플랫폼에서 전날 방문자 수와 현재 방문자 수 간의 자기 상관관계를 확인함으로써, 마케팅 활동이나 프로모션 등의 변화가 방문자 수에 미치는 영향을 이해할 수 있게 된다.

## 부분자기상관 함수 (Partial ACF, PACF)

-   **다른 시차의 영향을 제거한 후에 나타나는 자기상관을 보여준다.**
-   해당 시점과 주어진 시차 차이의 관계를 확인할 때, **중간에 있는 시차들의 영향을 배제한다.**
-   현재 시점을 기준으로 lag를 설정하면, 전날과의 차이를 계속 구해 나가는 것이 아니라 전전날, 전전전날 등 부분적으로 영향을 주는 시차를 확인할 수 있다.
-   이 때, 다른 시차의 영향을 제거하고 해당 시차와의 상관관계만 측정한다.
-   PACF를 통해 데이터의 **직접적인 상관관계를 파악하는 것은 유용하지만, 정상 시계열과 비정상 시계열을 구분하는 데에 활용하기 어렵다.**

            1. 특정 시차의 영향을 반복적으로 제거한다.
            2. 이전 lag와의 상관관계뿐만 아니라 훨씬 이전의 시차와의 상관관계도 측정할 수 있다.
            3. 직접적인 상관관계를 파악하는 데 유용하고, 특정 시차에 대한 자기상관을 직접적으로 보여준다.
            4. 이를 통해 어떤 시점이 다른 시점에 미치는 가에 대한 영향력을 파악하는데 용이하다.
            5. 주식 시장에서 주가 예측 모델을 구축하는 경우, 특정 시점의 주가와 한 달 전의 주가 간의 직접적인 상관관계를 확인함으로써 한 달 전의 주가 변동이 현재 주가에 미치는 영향을 파악할 수 있게 된다.

## 자기회귀 (AR, Autoregressive Model)

-   현 시점의 자료가 p시점 전의 유한개의 과거 자료로 설명될 수 있다.
-   과거의 데이터가 미래 데이터에 영향을 준다고 가정하는 모델이다.
-   현 시점의 시계열 자료에서 몇 번째 전 자료까지 영향을 주는 가를 파악하는 데에 사용된다.
-   현재 시점의 데이터가 직전 시점의 데이터에만 영향을 받는 모델을 1차 자기회귀 모형이라 하고, AR(1)로 표기한다.
-   이를 알아내기 위해서는 데이터의 패턴을 분석해야 하고, 이때 ACF, PACF를 사용한다.
-   표준정규분포 영역 내에 들어가는 첫 번째 지점을 절단점이라고 하며, 절단점에서 1을 빼준 값이 AR모델의 차수이다.

1. ACF

    > 만약 자기회귀 모델이라면,  
    > 현재 데이터와 멀리 떨어진 과거 데이터의 영향력은 점점 줄어들기 때문에 시간이 지남에 따라 상관관계가 줄어든다.

2. PACF
    > 만약 자기회귀 모델이라면,  
    > 특정 시점 이후에 급격히 감소하는 모양이 나타난다.

-   즉, 자기회귀 모델이라면, ACF는 시차가 증가함에 따라 점차 감소하고 PACF는 특정 시점 이후에 급격히 감소하여 절단된 형태를 보인다.
-   자기회귀 모델을 식별함으로써 데이터의 기본 패턴과 구조를 이해할 수 있고,
-   이를 통해 데이터가 어떻게 변동하는지, 과거 데이터가 미래에 어떤 영향을 미치는지에 대한 통찰력을 제공한다,

## 이동평균 (MA, Moving Average)

-   일정 기간 동안의 데이터를 평균하여 시계열 데이터의 부드러운 패턴(스무딩)을 확인할 수 있게 해준다.
-   특정 기간 동안의 데이터를 평균한 값으로, 시계열 데이터의 일정 기간의 평균을 보여준다.
-   데이터의 변동을 부드럽게 만들어서 패턴을 파악하는데 도움이 되며, 시계열 데이터의 추세를 이해하고 예측하는 데에 유용한 도구이다.

## 안정 시계열 (ARMA)

-   과거의 데이터와 최근의 평균을 사용하여 시계열 데이터의 패턴을 파악하고 예측하는 데에 사용한다.
-   2022년 3월 기준으로 ARMA가 중단되고 ARIMA로 대체되었다.
-   ARMA 모델은 시계열 데이터의 과거 값을 기반으로 한 선형 예측 모델이기 때문에
-   시계열 데이터가 정상성을 보이고, 예측에 영향을 주는 외부 요인이 없는 등의 가정을 만족해야 한다.
-   비정상성 데이터나 비선형적인 패턴을 갖는 데이터의 경우 패턴 파악 및 예측이 어렵다.
-   ARMA(1, 0) = AR(1)
-   ARMA(0, 1) = MA(1)
-   ARMA(1, 1) = AR(1), MA(1)

## <div id="ARIMA">불안정 시계열(ARIMA)</div>

-   ARIMA(p, d, q): d차 차분한 데이터에 AR(p)모형과 MA(q)모형을 합친 모델이다.
-   Autoregressive(자기 회귀), Integrated(누적 차분), Moving Average(이동평균)의 세 가지 요소로 구성되어 있다.
-   d는 비정상 시계열을 정상 시계열로 만들기 위해서 필요한 차분 횟수를 의미한다. (auto arima 활용)

<hr>

## 금융 시장 수익률 (Financial Market Return)

-   일반적으로 금융 시장을 분석하거나 머신러닝 모델을 구축할 때 price가 아닌 return을 활용하는 경우가 많다.
-   수익률을 통해 복리계산과 연율화(1년 간의 성장률)등을 위해 무수히 많은 반복 계산을 하게 되므로 (1 + return)으로 계산할 경우 복잡하고 번거로워진다.
-   기존 return에 log를 취해주면, 더 편한 계산이 가능해지지만 당연히 약간의 오차는 발생하게 된다.
-   약간의 오차를 감수하고 로그를 사용함으로써 얻은 이득이 훨씬 많고, 국내 주식에서의 상한선과 하한선은 -30% ~ 30% 제약까지 있기 때문에 일반적인 주가 움직임에 대해 오차가 극히 적다.
-   즉, 로그를 취하는 것은 정밀성보다 편의성을 높인 것으로 이해하면 된다.
-   수익률 단위가 분, 초, 밀리초 이하로 내려가야 정밀성에 차이가 많이 발생하지만, 실제 금융시장에서는 최소 하루 단위 이상으로 계산하기 때문에 오차가 거의 발생하지 않게 된다.
-   또한, return에 로그를 취하면 우측으로 치우친 확률 분포를 중심으로 재조정해주는 효과까지 있기 때문에 안쓸 이유가 없다.
-   골든 크로스, 데드 크로스 관련 및 내용에 대해서는 6번 코드 확인.

## <div id='Prophep'>Prophet (비트코인 시장 예측)</div>

-   페이스북에서 공계한 시계열 예측 라이브러리이다.
-   정확도가 높고 빠르며 직관적인 파라미터로 모델 수정이 용이하다는 장점이 있다.
-   Prophet 모델의 주요 구성요소는 Trend(주기적이지 않은 변화인 트랜드), Seasonality(주기적으로 나타나는 패턴 포함), Holiday(휴일과 같이 불규칙한 이벤트)
-   7번 코드를 참고할 것!
-   auto_arima의 경우 8번 코드를 참고할 것!!!!

## <div>Code Advanced</div>

<details>
    <summary>1. 시계열 데이터 가져와서 인덱스 설정 방법</summary>

    import yfinance as yf

    # 구글 주식 가져오기
    google_stock_df = yf.download('GOOG', start ='2014-05-01')
    google_stock_df

---

    # 인덱스 값이 datetime이 아닌 경우 변경해주는 함수 (reindex)
    google_stock_df = google_stock_df.reindex(pd.date_range(google_stock_df.index[0], google_stock_df.index[-1]))

</details>

<details>
    <summary>2. 차분 사용 방법 및 시각화 (diff(), pct_change())</summary>

    ## 연이은 관측값들의 차이를 계산한다(변화량).
    google_stock_df.diff()

---

    # 연이은 관측값들의 차이를 비율로 계산한다(변화율).
    google_stock_df.pct_change()

---

    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 5))

    plt.subplot(1, 3, 1)
    plt.plot(google_stock_df, c='orange')
    plt.title('Google stock price (2014~2024)')

    plt.subplot(1, 3, 2)
    plt.plot(google_stock_df.diff() )
    plt.title('Google stock differencing price (2014~2024)')

    plt.subplot(1, 3, 3)
    plt.plot(google_stock_df.pct_change())
    plt.title('Google stock differencing percent of price (2014~2024)')

    plt.show()

</details>

<details>
    <summary>3. 자기상관 함수 사용 및 정상 & 비정상 시계열 데이터 시각화(ACF)</summary>

    import numpy as np
    from statsmodels.tsa.stattools import acf
    import matplotlib.pyplot as plt

    # 데이터 프레임 전처리
    google_stock_df.dropna(inplace=True)
    # 1차 차분 데이터 프레임 생성
    google_stock_diff_df = google_stock_df.diff().dropna()

    # ACF 계산
    # google_stock_df: 비정상 시계열 데이터 프레임
    # google_stock_diff_df: 정상 시계열 데이터 프레임
    # nlags: 최대 시차 값을 저장
    google_stock_acf = acf(google_stock_df, nlags=20)
    google_stock_diff_acf = acf(google_stock_diff_df, nlags=20)

    # 플롯 설정
    plt.figure(figsize=(12, 5))

    # 비정상 시계열 ACF (차분을 진행하지 않은 데이터 프레임)
    plt.subplot(121)
    plt.plot(google_stock_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    # 이상치에 대한 부분을 표기하기 위해 +,- 1.96 사용하여 점선으로 표기. (유의미한 상관관계를 확인하기 위해).
    # 정상화가 되는 지에 대한 내용을 파악하기 위하여 사용.
    plt.axhline(y=-1.96 / np.sqrt(len(google_stock_df)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(google_stock_df)), linestyle='--', color='gray')
    plt.title('Non-Stationary Autocorrelation Function')

    # 정상 시계열 ACF (1차 차분을 진행하여 정상화 시킨 시계열 데이터 프레임)
    plt.subplot(122)
    plt.plot(google_stock_diff_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(google_stock_diff_df)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(google_stock_diff_df)), linestyle='--', color='gray')
    plt.title('Stationary Autocorrelation Function')

    # 플롯 출력
    plt.show()

---

    # acf 라이브러리를 사용하여 그래프 표기
    from statsmodels.graphics.tsaplots import plot_acf
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    plot_acf(google_stock_df, lags=20, ax=ax[0])
    plot_acf(google_stock_diff_df, lags=20, ax=ax[1])

    plt.tight_layout()
    plt.show()

</details>

<details>
    <summary>4. 비정상 시계열 MA/ 차분을 이용한 정상 시계열 MA 시각화 그래프</summary>
    
    import matplotlib.pyplot as plt

    plt.figure(figsize = (15, 20))


    plt.subplot(2,1,1)
    plt.plot(google_stock_df, c='orange')

    plt.subplot(2,1,1)
    plt.plot(moving_avg)

    plt.title('Google stock MR (2014~2024)')

    plt.subplot(2,1,2)
    plt.plot(google_stock_diff_df, c='orange')

    plt.subplot(2,1,2)
    plt.plot(moving_avg_diff)

    plt.title('Google stock differencing MR (2014~2024)')

    plt.show()

</details>

<details>
    <summary>5. ARIMA 사용 & (머신러닝 검증 평가) </summary>
    
    import pandas as pd
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(google_stock_df, order = (1, 1, 1))
    model_fit = model.fit()

    # 머신러닝에서는 전체를 훈련시켜서 그 기간내에 잘 맞추는 지 평가
    # 모델평가
    # 이유: 다음 차수를 알아야 다음을 알 수 있는데 1개 1개 업데이트를 쳐야 하기 때문에
    start_index = pd.to_datetime('2022-02-01')
    end_index = pd.to_datetime('2024-04-17')

    forecast = model_fit.predict(start=start_index, end=end_index)

    plt.figure(figsize=(15, 8))
    plt.plot(google_stock_df, label='original')
    plt.plot(forecast, label='predicted', c='orange')
    plt.title('Time Series Forecase')
    plt.show()

---

    from sklearn.metrics import mean_squared_error, mean_squared_log_error

    # 평균제곱오차 (MSE)
    mse = mean_squared_error(google_stock_df['2022-02-01':'2024-04-17'], forecast)
    print('Mean Squared Error', mse)

    # 루트 평균 제곱오차(RMSE): 예측값과 실제값의 차이를 제곱하여 평균을 구한 후 이를 다시 제곱근 하여 원래의 단위 변환
    mse = mean_squared_error(y_predict_df.test, y_predict_df.pred)
    print("Root Mean Squared Error", mse ** (1/2))

    # 평균 제곱 로그 오차(MSLE)
    msle = mean_squared_log_error(google_stock_df['2022-02-01':'2024-04-17'], forecast)
    print('Mean Squared Log Error', msle)

</details>

<details>
    <summary>6. 금융시장 수익률 시각화 및 골든, 데드 크로스 / auto_arima / </summary>

    # 수익률을 구히기 위해 shift 를 사용해서 이동 시켜서 값을 확인 한다
    # (이유: 수익률 => 오늘 수익값 / 어제 수익값 * 100)
    display(f_df.shift(1).head(4))
    display(f_df.head(4))
    display(f_df.shift(-1).head(4))

---

    import numpy as np

    # 수익률 df
    rate_f_df = np.log(f_df / f_df.shift(1))
    rate_f_df

---

    # 일간 수익률

    rate_f_df[['AAPL', 'MSFT', 'INTC', 'AMZN', 'SPY', 'GLD']].plot(figsize=(10, 5), lw=0.5)

---

    # 연율화
    # 연간 영업일(약 252일로 계산)
    # 252 = (통상적으로) 1년 영업률

    # 해당 코드는 log가 씌워져있는 상황
    rate_f_df.mean() * 252

---

    # 만약 단변량 데이터가 아닌 다변량 데이터 일 경우 다중 공선성에 대한 부분도 확인 하는 것이 도움이 될 것.
    import pandas as pd
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    def get_vif(features):
        vif = pd.DataFrame()
        vif["vif_score"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
        vif["feature"] = features.columns
        return vif

---

    rate_f_df = rate_f_df.dropna()
    get_vif(rate_f_df)

---

    # 각 원소들의 누적합 : cumsum()
    # 일간 수익률
    rate_f_df.cumsum().apply(np.exp).plot(figsize=(12, 6))
    plt.show()

    # 월간 수익률
    rate_f_df.cumsum().apply(np.exp).resample('1m').last().plot(figsize=(12, 6))
    plt.show()

---

    # 아마존 주식 시장 그래프 확인
    window = 20

    amzn_df['min'] = amzn_df['AMZN'].rolling(window=window).min()
    amzn_df['mean'] = amzn_df['AMZN'].rolling(window=window).mean()
    amzn_df['std'] = amzn_df['AMZN'].rolling(window=window).std()
    amzn_df['median'] = amzn_df['AMZN'].rolling(window=window).median()
    amzn_df['max'] = amzn_df['AMZN'].rolling(window=window).max()

    amzn_df = amzn_df.dropna()
    amzn_df

---

    import matplotlib.pyplot as plt

    ax = amzn_df[['min', 'mean', 'max']].iloc[-252:].plot(figsize= (12, 6), style=['g--','r--','g--'], lw=0.8)
    amzn_df['AMZN'].iloc[-252:].plot(ax=ax)
    plt.title("AMZN 20-Day Moving Average Price Movement")
    plt.show()

---

    # SMA(Simple Moving Average): 일정 기간동안의 가격의 평균을 나타내는 보조지표
    # 1달 영업일을 21일로 가정, 1년 영업일을 252일로 가정

    # 단기
    amzn_df['SMA1'] = amzn_df['AMZN'].rolling(window=21).mean() #short-term
    # 장기
    amzn_df['SMA2'] = amzn_df['AMZN'].rolling(window=252).mean() #long-term

    amzn_df[['AMZN', 'SMA1', 'SMA2']].tail()

---

    # 아마존 주가 기술 분석
    # 골든 크로스, 데드 크로스
    amzn_df.dropna(inplace=True)

    amzn_df['positions'] = np.where(amzn_df['SMA1'] > amzn_df['SMA2'], 1, -1)  # 1: buy , -1: sell /

    ax = amzn_df[['AMZN', 'SMA1', 'SMA2', 'positions']].plot(figsize=(15, 6), secondary_y='positions')
    ax.get_legend().set_bbox_to_anchor((-0.05, 1))

    plt.title("AMZN Trading Window based on Technical Analysis")
    plt.show()

</details>

<details>
    <summary>6-1. auto_arima 및 model 지표(MAPE)</summary>

    # 훈련 데이터와 검증 데이터 분리
    y_train = amzn_df['AMZN.O'][:int(0.8 * len(amzn_df))]
    y_test = amzn_df['AMZN.O'][int(0.8 * len(amzn_df)):]


    from pmdarima.arima import ndiffs
    # KPSS(Kaiatkowski-Phillips-Schmidt-Shin)
    # 차분을 진행하는 것이 필요할 지 결정하기 위해 사용하는 한 가지 검정 방법
    # 영가설(귀무가설)을 "데이터의 정상성이 나타난다."로 설정한 뒤
    # 영가설이 거짓이라는 증거를 찾는 알고리즘이다.
    # alpha: 강도, max_d: 최대 차분 횟수

    kpss_diffs = ndiffs(y_train, alpha=0.05, test='kpss', max_d=6)
    adf_diff = ndiffs(y_train, alpha=0.05, test='adf', max_d=6)
    pp_diff = ndiffs(y_train, alpha=0.05, test='pp', max_d=6)

    # 이 3개 중 최대값을 가져오면 된다.
    n_diffs = max(kpss_diffs, adf_diff, pp_diff)

    print(f"d = {n_diffs}")

---

    # auto_arima: AR, 차분 횟수, MA 차수 확인
    # auto_arima 함수는 ARIMA 모델의 최적의 매개변수를 자동으로 찾기 위한 함수입니다.
    # y: 훈련시킬 데이터 시계열
    # d: 차분 횟수 (시계열 데이터를 정상화하기 위해 필요)
    # start_p: AR(p)의 초기값 (AR 모델의 차수의 초기값)
    # max_p: AR(p)의 최대값 (AR 모델의 차수의 최대값)
    # start_q: MA(q)의 초기값 (MA 모델의 차수의 초기값)
    # max_q: MA(q)의 최대값 (MA 모델의 차수의 최대값)
    # m: 계절성을 띄는 경우 주기 (계절성 주기 설정)
    # seasonal: 계절성을 사용할지 여부 (True 또는 False)
    # stepwise: stepwise 알고리즘을 사용할지 여부 (True로 설정하면 자동으로 최적의 모델을 찾음)
    # trace: 최적화 과정을 출력할지 여부 (True로 설정하면 과정이 출력됨)

    model = pm.auto_arima(y = y_train,
                        d=1,
                        start_p=0,
                        max_p=3,
                        start_q=0,
                        max_q=3,
                        m=1,
                        seasonal=False,
                        stepwise=True,
                        trace=True)

---

    model.fit(y_train)

---

    # Prob(Q), 융-박스 검정 통계량
    # 영가설: 잔차가 백색잡음 시계열을 따른다. ** 백색잡음 : 분포를 잡을 수 있는 정도의 시계열
    # 0.05 이상: 서로 독립적이고 동일한 분포를 따른다.

    # Prob(H), 이분산성 검정 통계량
    # 영가설: 잔차기 이분산성을 띄지 않는다.
    # 0.05 이상: 잔차의 분산이 일정하다.

    # Prob(JB), 자크-베라 검정 통계량
    # 영가설: 잔차가 정규성을 따른다.
    # 0.05 이상: 일정한 평균과 분산을 따른다.

    # 이 3가지 검증을 가지고 금융 데이터에서 어떤 상품을 추천할 지 예측할 수 있음 (장기적 투가는 고위험, 단기 투자 권장) 이런식으로 작성

    # Skew(쏠린 정도, 왜도)
    # 0에 가까워야 한다.

    # Kurtosis(뾰족한 정도, 첨도)
    # 3에 가까워야 한다.

    print(model.summary())

    # N(0, 1) 정규분포

---

    import matplotlib.pyplot as plt

    model.plot_diagnostics(figsize=(16, 8))
    plt.show()

---

    # y_test를 아는 것 만큼 비교가 가능하기 때문에 predict 진행 시 주기를 적어준다.
    # 이건 잘못된 경우 : 다음 차수 1개에 대한 패턴으로 계속 신뢰구간이 증가되며
    # 정상적으로 할때는 다음 차수의 값을 결정한 후 추가 update를 해줘야 함

    # 현재는 유사한 양상으로 약 3 정도가 지속적으로 증가함.
    prediction = model.predict(len(y_test))
    prediction

---

    # 신뢰구간의 평균값이 예측값이다.
    prediction, conf_int = model.predict(n_periods =1, return_conf_int=True)
    print(conf_int)
    print(prediction)

---

    # 지속적으로 업데이트를 하기 위한 방법 (예측은 한발자국씩 진행되어야 함)
    prediction.tolist()[0]

---

    # 지속적으로 업데이트를 하기 위한 함수 선언. (예측은 한발자국씩 진행되어야 함)
    def predict_one_step():
        prediction = model.predict(n_periods =1)
        return (prediction.tolist()[0])

---

    # 예측된 횟수(step) 및 예측된 값을 통해서 model 을 지속적으로 update 하기 위한 반복문.
    preds = []
    p_list = []

    for data in y_test:
        p = predict_one_step()
        p_list.append(p)

        model.update(data)

---

    # test 데이터와 model이 예측한 데이터의 정도를 비교하기 위한 데이터 프레임 생성.
    y_predict_df = pd.DataFrame({"test": y_test, "pred": p_list})
    y_predict_df

---

    # auto_arima 를 이용한 데이터 시각화
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    plt.plot(y_train.iloc[-50:], label='Train')
    plt.plot(y_test.iloc[-50:], label='Test')
    plt.plot(y_predict_df.pred, label='Prediction')
    plt.legend()
    plt.show()

---

    # auto_arima 를 이용한 test 데이터 검증(MAPE).

    import numpy as np

    # MAPE (Mean Absolute Percentage Error): 에러의 절대값의 평균 함수 생성
    # 평균 절대 백분율 오차
    def MAPE(y_test, y_pred):
        return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f'MAPE (%): {MAPE(y_test, p_list):.4f}')

</details>

<details>
    <summary>7. Prophet (비트코인 분석)</summary>

    # 비트코인 데이터 가져오기
    import pandas as pd
    import json

    with open('./datasets/bitcoin_2010_2024.json') as f:
        json_data = json.load(f)

    bitcoin_df = pd.DataFrame(json_data['market-price'])
    bitcoin_df

---

    # 밀리초로 표기되어 있는 독립변수에 대하여 문자열(연-월-일 변경)
    from datetime import datetime

    def changeDate(milis):
        timestamp = milis / 1000
        converted_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        return converted_time

    # apply 함수 적용
    bitcoin_df.x = bitcoin_df.x.apply(changeDate)
    bitcoin_df

---

    # 독립변수로 존재하는 시계열 데이터 index로 변환
    bitcoin_df.set_index('x', inplace=True)
    bitcoin_df

---

    # 인덱스 번호 DatetimeIndex  로 변환
    bitcoin_df.index = pd.to_datetime(bitcoin_df.index)
    bitcoin_df.info()

---

    # prophet을 사용하기 위해서 시계열 인덱스를 독립변수로 변경 (reset_index())
    pre_b_df = bitcoin_df.reset_index()
    pre_b_df

---

    # prophet 모델을 사용하기 위해 시계열 인덱스를 ds 라는 독립변수로 선언해줘야 한다.
    pre_b_df = pre_b_df.rename(columns={'x': 'ds'})
    pre_b_df

---

    # Prophet 훈련
    from prophet import Prophet

    model = Prophet().fit(pre_b_df)

---

    # 향후 365일간의 예측을 위해 미래 데이터프레임 생성
    future = model.make_future_dataframe(periods=365)

    # 미래 데이터프레임을 사용하여 예측 수행
    forecast = model.predict(future)

    # 실제로 예측한 값이 정확하지 않을 수 있으며 그렇기 때문에 신뢰구간을 주의 깊게 봐야한다.
    # yhat: 예측한 값
    # yhat_lower, yhat_upper: 예측 신뢰 구간의 하한과 상한
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

---

    # 실제 값과 향후 예측한 모델에 대한 시각화
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    plt.plot(pre_b_df[['y']], label='Train')
    plt.plot(forecast[['yhat']], label='Prediction')
    plt.legend()
    plt.show()

---

    # 실제 값과 향후 예측한 모델에 신뢰 구간이 포함된 대한 시각화
    model.plot(forecast, figsize=(15, 8), xlabel='year-month', ylabel='price')
    plt.show()

---

    # 추세, 요일, 월간 시각화 그래프 (옵션 사용 시 일간, 공유일 효과 추가 할 수 있음)
    model.plot_components(forecast, figsize=(20, 20))
    plt.show()

</details>

<details>
    <summary>9. Prophet-하이퍼 파라미터 조절!! (비트코인 분석)</summary>
    
    # train 데이터프레임을 사용하여 분석 후 test 데이터프레임을 사용하여 오차 확인 및 예측 진행
    # 80% train 데이터 분리
    train_df = pre_b_df.iloc[:int(len(pre_b_df) * 0.8)]
    train_df

---

    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    # itertools : 각 요소에 대하여 경우의 수 별로 묶어 줄 수 있도록 사용되는 라이브러리
    import itertools

    # changepoint_prior_scale: trend의 변화하는 크기를 반영하는 정도이다, 0.05가 default
    # seasonality_prior_scale: 계절성을 반영하는 단위이다.
    # seasonality_mode: 계절성으로 나타나는 효과를 더해 나갈지, 곱해 나갈지 정한다.

    # 왠만해서는 해당 수치로 분석할 것
    search_space = {
        'changepoint_prior_scale': [0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        'seasonality_prior_scale': [0.05, 0.1, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }

    # itertools.product(): 각 요소들의 모든 경우의 수 조합으로 생성

    param_combinded = [dict(zip(search_space.keys(), v)) for v in itertools.product(*search_space.values())]

    # 데이터셋 분할 후 길이 확인: 80% 훈련 데이터, 20% 테스트 데이터
    train_len = int(len(pre_b_df) * 0.8)
    test_len = int(len(pre_b_df) * 0.2)

    # 평가 시 dats 라는 문구가 붙어야 하기 때문에 f-string을 이용한 구문 생성
    train_size = f'{train_len} days'
    test_size = f'{test_len} days'
    train_df = pre_b_df.iloc[: train_len]
    test_df = pre_b_df.iloc[train_len: ]

     모델의 성능 평가를 위한 MAPE 값을 저장할 리스트
    mapes = []
    for param in param_combinded:
        model = Prophet(**param)
        model.fit(train_df)

            # 'threads' 옵션은 메모리 사용량은 낮지만 CPU 바운드 작업에는 효과적이지 않을 수 있다.
            # 'dask' 옵션은 대규모의 데이터를 처리하는 데 효과적이다.
            # 'processes' 옵션은 각각의 작업을 별도의 프로세스로 실행하기 때문에 CPU 바운드 작업에 효과적이지만,
            # 메모리 사용량이 높을 수 있다.

            # 모델의 교차검증을 수행하여 모델의 예측 성능 평가하는데 사용
            # 일반화 성능을 평가
            # initial: 초기 훈련 데이터의 기간, period: 예측을 수행할 간격, horizon: 훈련 단계 후 예측할 기간, parallel: 병렬 처리 방식
            cv_df = cross_validation(model, initial=train_size, period='20 days', horizon=test_size, parallel='processes')

            # 모델의 성능을 평가하여 MAPE 값을 계산
            # performance_metrics: 모델의 성능을 평가하는 메소드
            df_p = performance_metrics(cv_df, rolling_window=1)
            mapes.append(df_p['mape'].values[0])

    # 튜닝 결과를 데이터프레임으로 저장
    tuning_result = pd.DataFrame(param_combinded)
    tuning_result['mape'] = mapes

---

    # 최적의 튜닝 결과를 확인
    tuning_result.sort_values(by='mape')

---

    # 최적의 하이퍼파라미터 값으로 model을 다시 훈련 시켜서 값을 확인
    model = Prophet(changepoint_prior_scale=0.5,
                    seasonality_prior_scale=0.1,
                    seasonality_mode='multiplicative')

    model.fit(pre_b_df)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][221:]

---

    # 시계열 데이터프레임으로 다시 만들어 시각화를 편하게 하기위하여 인덱스 재정의 후 데이터 프레임 재선언
    b_df = pre_b_df.copy()
    b_df.set_index('ds', inplace=True)

    # 예측 결과 데이터 프레임 생성
    forecast_df = forecast.copy()
    forecast_df = forecast_df.set_index('ds')

    b_df.index = pd.to_datetime(b_df.index)
    forecast_df.index = pd.to_datetime(forecast_df.index)

    # 시각화
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    plt.plot(b_df[['y']], label='Train')
    plt.plot(forecast_df[['yhat']], label='Prediction')
    plt.legend()
    plt.show()

---

    model.plot(forecast, figsize=(15, 8), xlabel='year-month', ylabel='price')
    plt.show()

    model.plot_components(forecast, figsize=(20, 20))
    plt.show()

</details>
