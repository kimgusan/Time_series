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

## <div id='Prophep'>Prophet (비트코인 시장 예측)</div>

-   페이스북에서 공계한 시계열 예측 라이브러리이다.
-   정확도가 높고 빠르며 직관적인 파라미터로 모델 수정이 용이하다는 장점이 있다.
-   Prophet 모델의 주요 구성요소는 Trend(주기적이지 않은 변화인 트랜드), Seasonality(주기적으로 나타나는 패턴 포함), Holiday(휴일과 같이 불규칙한 이벤트)
-   6번 코드를 참고할 것!

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
    <summary>6. Prophet (비트코인 분석)</summary>

</details>
