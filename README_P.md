# 💹 리튬 ETF 주가 확인
- 2024년 06월 14일 기준

## 목차

1. 데이터 시각화 및 분석
2. 데이터 전처리
3. 모델 평가
4. 딥러닝(Prophet) 모델 사용 및 예측
5. 에측 결과 분식 및 설명
6. 느낀점

---

### 1. 데이터 시각화 및 분석

<details>
    <summary>데이터 불러오기</summary>
    
    import yfinance as yf
    import pandas as pd

    # 금 ETF
    gold_ticker = 'GLD'
    g_df = yf.download(gold_ticker, start='2013-01-01')['Adj Close'].round(4)

    # 리튬 ETF
    lithium_ticker = 'LIT'
    l_df = yf.download(lithium_ticker, start='2013-01-01')['Adj Close'].round(4)

    # 데이터프레임 합치기
    c_df = pd.DataFrame({
        'Gold': g_df,
        'Lithium': l_df
    })

    # 데이터프레임 확인
    c_df.head()

</details>

#### 1-1. 금 ETF과 리튬 ETF 시계열 데이터 유형 확인.

<img width="993" alt="스크린샷 2024-06-17 오후 5 08 11" src="https://github.com/kimgusan/Time_series/assets/156397911/bc9087f2-ab7d-4ba0-8b79-f46708fe6076">

> 2013년 01월 ~ 2024년 06월 14일 까지의 그래프를 봤을 때 계절성이나 별도의 추세는 보이지 않는 그래프를 확인.  
> 이때, 리튬 베터리의 그래프가 2020년 부터 급격하게 반등한 부분을 확인할 수 있음.

### 2. 데이터 전처리 (ACF, PACF 등)

<details>
    <summary>ACF, PACF 시각화 코드</summary>

    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    # 그래프의 행, 열 및 크기 조절
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    # l_df에 대한 ACF와 PACF 플롯
    plot_acf(l_df, lags=20, ax=ax[0][0])
    ax[0][0].set_title('ACF of Lithium')

    plot_pacf(l_df, lags=20, ax=ax[0][1])
    ax[0][1].set_title('PACF of Lithium')

    # g_df에 대한 ACF와 PACF 플롯
    plot_acf(g_df, lags=20, ax=ax[1][0])
    ax[1][0].set_title('ACF of Gold')

    plot_pacf(g_df, lags=20, ax=ax[1][1])
    ax[1][1].set_title('PACF of Gold')

    # 레이아웃 조절
    plt.tight_layout()
    plt.show()

</details>

#### 2-1. 금 ETF과 리튬 ETF 시계열 그래프 유형 확인(ACF, PACF 활용).

<img width="876" alt="스크린샷 2024-06-17 오후 5 09 55" src="https://github.com/kimgusan/Time_series/assets/156397911/b6efbb22-bb03-4c0c-b76a-e1f73dc4189d">

<img width="876" alt="스크린샷 2024-06-17 오후 5 12 28" src="https://github.com/kimgusan/Time_series/assets/156397911/302dc8fd-aedd-4404-b8e0-5d7691643db5">

> 금과 리튬 ETF 시계열 데이터 모두 정상성을 띄지 않는 비정상 시계열 데이터임을 확인.  
> 이후 모델 평가 진행 시 차분을 통한 데이터 유형 변경 필요.

#### 2-2. 금 ETF과 리튬 ETF 변화율 확인.

<img width="430" alt="스크린샷 2024-06-17 오후 5 22 38" src="https://github.com/kimgusan/Time_series/assets/156397911/df26f566-b06d-4610-9ea0-38a09d4fe981">

> 금보다는 리튬 ETF의 변화율 큰 부분 확인.

<details>
    <summary>금 ETF과 리튬 ETF 시계열 데이터 변화율 및 수익률 확인.</summary>

    # 변화율! 확인
    c_df.pct_change().mean().plot(kind='bar',figsize=(5,5), grid=True)
    plt.xticks(rotation=45)
    plt.show()

---

    import numpy as np

    # 수익률 df
    # 수익률의 경우 다음날과 비교하여 전날의 수익률을 나눠야 하기 때문에 해당 공식을 사용
    rate_c_df = np.log(c_df / c_df.shift(1))
    rate_c_df

---

    import numpy as np


    # 일간 수익률
    fig, ax = plt.subplots(figsize=(15, 7))
    rate_c_df['Gold'].plot(ax=ax, lw=0.7, color='blue', label='Gold')
    rate_c_df['Lithium'].plot(ax=ax, lw=0.7, color='red', alpha=0.5, label='Lithium')  # 투명도 조절
    ax.legend()
    plt.show()

---

    # 각 원소들의 누적합 : cumsum()
    # 일간 수익률
    rate_c_df.cumsum().apply(np.exp).plot(figsize=(12, 6))
    plt.show()

</details>

#### 2-3. 금 ETF과 리튬 ETF 다중공산성 확인.

<img width="170" alt="스크린샷 2024-06-17 오후 5 17 01" src="https://github.com/kimgusan/Time_series/assets/156397911/86d4226d-69dc-4762-8af9-26eb1a1b7a71">

> 금 ETF과 리튬 ETF에 대해서는 다중공산성이 존재 하지 않는 것으로 확인.

#### 2-4. 금 ETF과 리튬 ETF 일간 & 월간 수익률 확인.

<img width="894" alt="스크린샷 2024-06-17 오후 5 26 10" src="https://github.com/kimgusan/Time_series/assets/156397911/c5d6f557-4e77-4a52-bff8-0584c7ef5608">
<img width="894" alt="스크린샷 2024-06-17 오후 5 27 15" src="https://github.com/kimgusan/Time_series/assets/156397911/002537a5-82ca-48c4-bac6-95f6c0306a2a">
<img width="884" alt="스크린샷 2024-06-17 오후 5 27 27" src="https://github.com/kimgusan/Time_series/assets/156397911/8cd68c80-3287-4b0c-9a10-878a65a04848">

> 리튬 ETF 의 경우 특정 시기에 높은 수익율을 보이고 있지만, 단발성인 부분 확인.

#### 2-5. 리튬 ETF 차분 진행 시 정상성을 띄는 그래프 확인.

<img width="978" alt="스크린샷 2024-06-17 오후 5 21 19" src="https://github.com/kimgusan/Time_series/assets/156397911/bc3e7e12-bd9d-4cf3-be4f-bb43a0231265">  
> 해당 데이터를 사용하기 위해 비정상 시계열에서 차분을 사용하여 정상 시계열 데이터로 변경하였을 때 어느정도 부분이 나타나는 것을 확인.

#### 2-6. 리튬 ETF에 대하여 SMA를 적용하여 골든 크로스, 데드크로스 확인.

<img width="995" alt="스크린샷 2024-06-17 오후 5 28 50" src="https://github.com/kimgusan/Time_series/assets/156397911/70774076-8beb-408a-9608-708f4e92e47e">
<img width="995" alt="스크린샷 2024-06-17 오후 5 30 15" src="https://github.com/kimgusan/Time_series/assets/156397911/5b4b3648-e37a-44e1-8905-acda9f1e25b7">

> 2020년도에 리튬 ETF가 2020년 이후 크게 상승한 부분이 있지만 2022년 이후에 하락세를 보이고 있으며 2020년 이전 분포 까지 내려가려는 추세 확인.

<details>
    <summary>골든 크로스, 데드 크로스 그래프 </summary>

    window = 20

    pre_l_df['min'] = pre_l_df['Lithium'].rolling(window=window).min()
    pre_l_df['mean'] = pre_l_df['Lithium'].rolling(window=window).mean()
    pre_l_df['std'] = pre_l_df['Lithium'].rolling(window=window).std()
    pre_l_df['median'] = pre_l_df['Lithium'].rolling(window=window).median()
    pre_l_df['max'] = pre_l_df['Lithium'].rolling(window=window).max()

    pre_l_df = pre_l_df.dropna()
    pre_l_df

    import matplotlib.pyplot as plt

---

    ax = pre_l_df[['min', 'mean', 'max']].iloc[-252:].plot(figsize= (12, 6), style=['g--','r--','g--'], lw=0.8)
    pre_l_df['Lithium'].iloc[-252:].plot(ax=ax)
    plt.title("Lithium 1-year Moving Average Price Movement")
    plt.show()

---

    # SMA(Simple Moving Average): 일정 기간동안의 가격의 평균을 나타내는 보조지표
    # 1달 영업일을 21일로 가정, 1년 영업일을 252일로 가정

    # 단기
    pre_l_df['SMA1'] = pre_l_df['Lithium'].rolling(window=21).mean() #short-term
    # 장기
    pre_l_df['SMA2'] = pre_l_df['Lithium'].rolling(window=252).mean() #long-term

    pre_l_df[['Lithium', 'SMA1', 'SMA2']].tail()

---

    # 골든 크로스, 데드 크로스 확인

    pre_l_df.dropna(inplace=True)

    pre_l_df['positions'] = np.where(pre_l_df['SMA1'] > pre_l_df['SMA2'], 1, -1)  # 1: buy , -1: sell /

    ax = pre_l_df[['Lithium', 'SMA1', 'SMA2', 'positions']].plot(figsize=(15, 6), secondary_y='positions')
    ax.get_legend().set_bbox_to_anchor((-0.05, 1))

    plt.title("Lithium Trading Window based on Technical Analysis")
    plt.show()

</details>

### 3. 모델 평가 (ARIMA)

#### 3-1. KPSS, ADF, PP 테스트를 통한 차분 횟수 확인

> 차분 횟수: 2

<details>
    <summary>PSS, ADF, PP 테스트를 통한 차분 횟수 확인.</summary>

    from pmdarima.arima import ndiffs

    # KPSS(Kwiatkowski-Phillips-Schmidt-Shin) 테스트를 통해 차분이 필요한 횟수 계산
    # alpha=0.05: 유의수준 5%, max_d=6: 최대 차분 횟수는 6
    kpss_diffs = ndiffs(y_train, alpha=0.05, test='kpss', max_d=6)

    # ADF(Augmented Dickey-Fuller) 테스트를 통해 차분이 필요한 횟수 계산
    # alpha=0.05: 유의수준 5%, max_d=6: 최대 차분 횟수는 6
    adf_diff = ndiffs(y_train, alpha=0.05, test='adf', max_d=6)

    # PP(Phillips-Perron) 테스트를 통해 차분이 필요한 횟수 계산
    # alpha=0.05: 유의수준 5%, max_d=6: 최대 차분 횟수는 6
    pp_diff = ndiffs(y_train, alpha=0.05, test='pp', max_d=6)

    # 위의 세 테스트에서 나온 차분 횟수 중 최대값을 선택
    n_diffs = max(kpss_diffs, adf_diff, pp_diff)

    # 최종 차분 횟수를 출력
    print(f"d = {n_diffs}")

</details>

#### 3-2. auto_arima를 활용한 AR 차수, MA 차수, 차분 횟수 확인.

<div style="display: flex; justify-content: space-around;">
  <div>
    <img src="https://github.com/kimgusan/Time_series/assets/156397911/a9f785bf-3197-4402-bff3-f46823c08774" alt="Image 1" style="width: 300px;">
  </div>
  <div>
    <img src="https://github.com/kimgusan/Time_series/assets/156397911/e0a6f856-149e-4611-a298-2293606a6189" alt="Image 2" style="width: 300px;">
  </div>
</div>


<details>
    <summary>auto_arima 사용</summary>

    import pmdarima as pm
    model = pm.auto_arima(y = y_train,
                        d = 2,
                        start_p =0,
                        max_p = 4,
                        start_q = 0,
                        max_q = 21,
                        m=1,
                        seasonal = False,
                        stepwise = True,
                        trace = True
                        )


    # y: 학습에 사용할 시계열 데이터입니다. 이 데이터는 종속변수로 사용됩니다.
    # d: 차분 횟수입니다. 차분은 시계열 데이터를 안정화하는 데 사용됩니다. 여기서는 2번 차분을 수행합니다.
    # start_p와 max_p: AR(자기회귀) 모델의 차수 범위를 설정합니다. start_p는 최소 차수를, max_p는 최대 차수를 나타냅니다. 이 범위 내에서 최적의 차수를 찾습니다.
    # start_q와 max_q: MA(이동평균) 모델의 차수 범위를 설정합니다. start_q는 최소 차수를, max_q는 최대 차수를 나타냅니다. 이 범위 내에서 최적의 차수를 찾습니다.
    # m: 계절성을 나타내는 주기입니다. 계절성이 없는 경우 m=1로 설정합니다. 예를 들어, 월간 데이터의 경우 m=12로 설정할 수 있습니다.
    # seasonal: 계절성 ARIMA 모형을 사용할지 여부를 나타냅니다. 계절성을 사용하려면 True로 설정하고, 계절성을 사용하지 않으려면 False로 설정합니다.
    # stepwise: 단계별로 최적의 모형을 찾을지 여부를 나타냅니다. True로 설정하면 단계별로 최적의 모형을 찾습니다. 이 방법은 계산 속도를 높일 수 있습니다.
    # trace: 모델 학습 과정을 출력할지 여부를 나타냅니다. True로 설정하면 학습 과정을 출력하여 모델 선택 과정을 확인할 수 있습니다.

    model.fit(y_train)

    model.summary()

</details>

#### 3-3. auto_arima 모델에 대하여 SARIMAX Results 확인 및 평가.

<img width="613" alt="스크린샷 2024-06-17 오후 5 34 23" src="https://github.com/kimgusan/Time_series/assets/156397911/b85a96c7-9fdd-4ad4-bd87-6574bc967ef7">

> 추가 통계량 해석

    1. Ljung-Box (L1) (Q)
    - Ljung-Box (L1) (Q): 3.74, Prob(Q): 0.05: Ljung-Box 검정은 잔차가 백색잡음인지 확인하는 테스트입니다.
    p-값이 0.05로 나타나며, 이는 잔차가 서로 독립적일 가능성이 있는 것으로 나타납니다.. 

    2. Jarque-Bera (JB)
    - Jarque-Bera (JB): 8174.96, Prob(JB): 0.00: Jarque-Bera 검정은 잔차의 정규성을 확인하는 테스트입니다.
    p-값이 0.00으로 매우 낮아, 잔차가 정규분포를 따르지 않음을 나타냅니다.

    3. Heteroskedasticity (H)
    Heteroskedasticity (H): 21.64, Prob(H): 0.00: 이분산성을 확인하는 테스트입니다.
    p-값이 0.00으로 매우 낮아, 잔차의 분산이 일정하지 않음을 나타냅니다.

    4. Skew: 0.14: 왜도를 나타냅니다. 값이 0에 가까울수록 대칭 분포를 나타내며, 양수값은 꼬리가 왼쪽으로 긴 분포를 의미합니다.

    5. Kurtosis: 12.22: 첨도를 나타냅니다. 값이 3에 가까울수록 정규분포에 가깝습니다. 높은 값은 분포가 뾰족함을 의미합니다.

    요약
    - Ljung-Box 검정 (Prob(Q)): 잔차가 완전히 독립적이지 않을 가능성을 시사합니다.
    - Jarque-Bera 검정 (Prob(JB)): 잔차가 정규성을 따르지 않습니다.
    - Heteroskedasticity 검정 (Prob(H)): 잔차의 분산이 일정하지 않습니다.
    - Skew and Kurtosis: 왜도는 거의 대칭적으로 판단되지만, 첨도는 매우 높아 분포가 뾰족합니다.

    결과
    - 장기 투자: 높은 Kurtosis와 낮은 Skewness는 데이터가 일정하지 않고 변동성이 클 수 있음을 시사합니다. 장기 투자는 고위험일 수 있습니다.
    - 단기 투자: 잔차가 백색잡음 분포를 따르므로, 단기적으로는 예측 가능성이 높아 단기 투자가 더 적합할 수 있습니다.
    이러한 해석을 바탕으로 모델의 적합성과 예측의 신뢰성을 평가할 수 있습니다.

<img width="979" alt="스크린샷 2024-06-17 오후 5 39 05" src="https://github.com/kimgusan/Time_series/assets/156397911/ac833e27-1acd-472a-96c3-d64045f013c6">

<img width="888" alt="스크린샷 2024-06-17 오후 10 51 58" src="https://github.com/kimgusan/Time_series/assets/156397911/0783018f-ac27-4cee-b1d9-c6152ee1d682">

<details>
    <summary>auto_arima 사용 시 단계별 예측 수행 코드</summary>

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
    # 지속적으로 업데이트를 하기 위한 방법 (예측은 한발자국씩 진행되어야 함)
    def predict_one_step():
        prediction = model.predict(n_periods =1)
        return (prediction.tolist()[0])
---
    preds = []
    p_list = []
    
    for data in y_test:
        p = predict_one_step()
        p_list.append(p)
    
        model.update(data)
---
    y_predict_df = pd.DataFrame({"test": y_test, "pred": p_list})
    y_predict_df
    
</details>

> MAPE (%) : 1.6601


#### 3-4. ARIMA 모델 평가.

<img width="993" alt="스크린샷 2024-06-17 오후 10 59 55" src="https://github.com/kimgusan/Time_series/assets/156397911/a5482310-47d0-417c-bbd3-a7f51736c685">

<details>
    <summary>ARIMA 모델을 사용하여 모델 평가</summary>
    from statsmodels.tsa.arima.model import ARIMA
    
    model = ARIMA(pre_l_df, order=(4, 2, 0))
    model_fit = model.fit()
    
    start_index = pd.to_datetime('2013-01-02')
    end_index = pd.to_datetime('2024-06-14')
    
    # 이 부분은 추가적인 예측이 아니라 기존 실제 데이터에서 모델 평가를 하는 부분
    forecast = model_fit.predict(start=start_index, end=end_index)
    
    plt.figure(figsize=(15, 8))
    
    # 실제 시계열 데이터
    plt.plot(pre_l_df['2021':], label='original')
    # model을 훈련시켜서 나온 결과에 대한 모델 검증
    plt.plot(forecast['2021':], label='predicted', c='orange')
    plt.title("Time Series Forecast (2023.01 ~ 2024.06.14)")
    plt.legend()
    plt.show()
    
</details>

> 모델 평가를 진행했을 때 분포는 준수한 성능을 보이는 것으로 판단.  
> Mean Squared Error 1.11  
> Root Mean Squared Error 1.05  
> Mean Squared Log Error 0.004


<details>
    <summary>ARIMA 모델을 사용한 리튬 ETF 평가</summary>

    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(pre_l_df, order=(4, 2, 0))
    model_fit = model.fit()

    start_index = pd.to_datetime('2013-01-02')
    end_index = pd.to_datetime('2024-06-14')

    # 이 부분은 추가적인 예측이 아니라 기존 실제 데이터에서 모델 평가를 하는 부분
    forecast = model_fit.predict(start=start_index, end=end_index)

    plt.figure(figsize=(15, 8))

    # 실제 시계열 데이터
    plt.plot(pre_l_df['2021':], label='original')
    # model을 훈련시켜서 나온 결과에 대한 모델 검증
    plt.plot(forecast['2021':], label='predicted', c='orange')
    plt.title("Time Series Forecast")
    plt.legend()
    plt.show()

    plt.figure(figsize=(15, 8))

---

    # 실제 시계열 데이터
    plt.plot(pre_l_df['2024':], label='original')
    # model을 훈련시켜서 나온 결과에 대한 모델 검증
    plt.plot(forecast['2024':], label='predicted', c='orange')
    plt.title("Time Series Forecast")
    plt.legend()
    plt.show()

</details>

### 4. 딥러닝 (Prophet) 모델 사용 및 예측

#### 4-1. Prophet 모델을 사용하기 위한 전처리

<img width="190" alt="스크린샷 2024-06-17 오후 11 09 12" src="https://github.com/kimgusan/Time_series/assets/156397911/cc7784e9-1f32-4171-8bc9-6770d3498e9c">


<details>
    <summary>날짜 인덱스 독립변수 선언 및 컬럼명 변경</summary>

    # prophet 모델을 사용하기 위해 시계열 인덱스를 ds 라는 독립변수로 선언해줘야 한다.
    pre_l_df = pre_l_df.rename(columns={'Date': 'ds'})
    pre_l_df = pre_l_df.rename(columns={'Lithium': 'y'})
    pre_l_df

</details>

#### 4-2. Prophet 파라미터 조정 없이 default 값으로 훈련 진행

<img width="887" alt="스크린샷 2024-06-17 오후 11 10 35" src="https://github.com/kimgusan/Time_series/assets/156397911/6cec2be2-d2fe-42d5-9ea6-ac947794b30c">

> 2020 년 이후 신뢰구간을 실측값이 신뢰구간을 벗어나는 부분 확인하여 파라미터 조정하는 방향으로 진행.

<details>
    <summary>Prophet fit Code_Cycle01</summary>

    from prophet import Prophet

    model = Prophet().fit(pre_l_df)

---

    # model를 통해 예측한 1년 결과값 생성
    future = model.make_future_dataframe(periods=365)

    # 1년 결과 예측
    forecast = model.predict(future)

    # 실제로 예측한 값이 정확하지 않을 수 있으며 그렇기 때문에 신뢰구간을 주의 깊게 봐야한다.
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

---

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    plt.plot(pre_l_df[['y']], label='Train')
    plt.plot(forecast[['yhat']], label='Prediction')
    plt.legend()
    plt.show()

---

    model.plot(forecast, figsize=(15, 8), xlabel='year-month', ylabel='price')
    plt.show()

</details>

#### 4-3. Prophet 파라미터 조정 후 훈련 진행.
예측 성능 지표 계산, 교차검증을 이용한 파라미터 확인

<img width="532" alt="스크린샷 2024-06-17 오후 11 23 21" src="https://github.com/kimgusan/Time_series/assets/156397911/2fda7066-3548-4f83-be02-6ec6e5711c0e">
<img width="898" alt="스크린샷 2024-06-17 오후 11 19 42" src="https://github.com/kimgusan/Time_series/assets/156397911/c10d92a1-1e8d-4f35-8e0a-acc4a682f479">
<img width="975" alt="스크린샷 2024-06-17 오후 11 19 50" src="https://github.com/kimgusan/Time_series/assets/156397911/afa5e850-dc8b-4ae0-8c1a-074188204ba3">

> 이전 모델과 비교하였을 때 실측값이 조금 더 신뢰구간에 가까워 진 부분 확인.  
> 예측 결과가 비정상적으로 불안정한 부분 확인, seasonality_mode를 변경하여 덧셈 방식으로 변경 후 추가 훈련 진행.
> 데이터의 특성상 계절성 또는 주기성이 잘 나타나지 않는 경우, 데이터의 변화가 일정한 비율로 증가하거나 감소하지 않으면  multiplicative 모델이 적합하지 않음.

-   changepoint_prior_scale=1,
-   seasonality_prior_scale=10,
-   seasonality_mode='multiplicative'
-   mape = 0.446866

<details>
    <summary>파라미터 확인 (cross_validation, performance_metrics)</summary>

    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    import itertools

    # changepoint_prior_scale: trend의 변화하는 크기를 반영하는 정도이다, 0.05가 default
    # seasonality_prior_scale: 계절성을 반영하는 단위이다.
    # seasonality_mode: 계절성으로 나타나는 효과를 더해 나갈지, 곱해 나갈지 정한다.
    search_space = {
        'changepoint_prior_scale': [0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        'seasonality_prior_scale': [0.05, 0.1, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }

    # itertools.product(): 각 요소들의 모든 경우의 수 조합으로 생성
    param_combinded = [dict(zip(search_space.keys(), v)) for v in itertools.product(*search_space.values())]

    train_len = int(len(pre_l_df) * 0.8)
    test_len = int(len(pre_l_df) * 0.2)

    train_size = f'{train_len} days'
    test_size = f'{test_len} days'
    train_df = pre_l_df.iloc[: train_len]
    test_df = pre_l_df.iloc[train_len: ]

    mapes = []
    for param in param_combinded:
        model = Prophet(**param)
        model.fit(train_df)

        # 'threads' 옵션은 메모리 사용량은 낮지만 CPU 바운드 작업에는 효과적이지 않을 수 있다.
        # 'dask' 옵션은 대규모의 데이터를 처리하는 데 효과적이다.
        # 'processes' 옵션은 각각의 작업을 별도의 프로세스로 실행하기 때문에 CPU 바운드 작업에 효과적이지만,
        # 메모리 사용량이 높을 수 있다.
        cv_df = cross_validation(model, initial=train_size, period='20 days', horizon=test_size, parallel='processes')
        df_p = performance_metrics(cv_df, rolling_window=1)
        mapes.append(df_p['mape'].values[0])

    tuning_result = pd.DataFrame(param_combinded)
    tuning_result['mape'] = mapes

---

    # 최적의 파라미터 확인
    tuning_result.sort_values(by='mape')

---

    # 최적의 파라미터 값으로 model을 다시 훈련 시켜서 값을 확인
    model = Prophet(changepoint_prior_scale=10,
                    seasonality_prior_scale=10,
                    seasonality_mode='multiplicative')

    model.fit(pre_l_df)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][221:]

---

    # 시계열 데이터프레임으로 다시 만들어 시각화를 편하게 하기위하여 인덱스 재정의 후 데이터 프레임 재선언
    reset_l_df = pre_l_df.copy()
    reset_l_df.set_index('ds', inplace=True)

    # 예측 결과 데이터 프레임 생성
    forecast_df = forecast.copy()
    forecast_df = forecast_df.set_index('ds')

    reset_l_df.index = pd.to_datetime(reset_l_df.index)
    forecast_df.index = pd.to_datetime(forecast_df.index)

---

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    plt.plot(reset_l_df[['y']], label='Train')
    plt.plot(forecast_df[['yhat']], label='Prediction')
    plt.legend()
    plt.show()

---

    model.plot(forecast, figsize=(15, 8), xlabel='year-month(2013.01 ~ 2025.06)', ylabel='price')
    plt.show()

    model.plot_components(forecast, figsize=(20, 20))
    plt.show()

</details>

#### 4-4. 데이터 훈련 날짜 조정 후 훈련 진행.

<img width="877" alt="스크린샷 2024-06-17 오후 11 26 09" src="https://github.com/kimgusan/Time_series/assets/156397911/240f2321-9391-4222-94ff-9f3cb932d313">
<img width="994" alt="스크린샷 2024-06-17 오후 11 26 32" src="https://github.com/kimgusan/Time_series/assets/156397911/7ade2071-8278-4574-9d59-20f0b14c5690">
<img width="982" alt="스크린샷 2024-06-17 오후 11 26 42" src="https://github.com/kimgusan/Time_series/assets/156397911/2637e392-5b9d-4d89-94b1-fd3b440449a4">
<img width="991" alt="스크린샷 2024-06-17 오후 11 26 50" src="https://github.com/kimgusan/Time_series/assets/156397911/94e54e9c-9b14-468a-b322-25fa9fe234ed">
<img width="983" alt="스크린샷 2024-06-17 오후 11 27 00" src="https://github.com/kimgusan/Time_series/assets/156397911/5fb6bc10-0257-4d6a-8dc9-ce70d7efce8c">



-   2020년부터 훈련 진행
-   multiplicative -> additive 변경
    (누적값이 아닌 곱셈 값으로 했을 때 예측 결과가 이상한 부분 확인)
> Multiplicative 방식이 적용될 경우, 데이터의 값이 크면 계절성 효과도 그만큼 크게 적용되어, 예측 값의 변동폭이 심해집니다.  
> 반대로, Additive 방식은 데이터의 값이 크더라도 계절성 효과가 일정하게 유지되어 예측 값의 변동폭이 안정적으로 유지됩니다.
-   changepoint_prior_scale=0.05,
-   seasonality_prior_scale=10,
-   seasonality_mode='additive'
-   mape = 0.117238 (기존 multiplicative 사용했을 때 보다 약 0.3 수치가 감소)
> pre_l_df = pre_l_df['2020':].reset_index()
- 신뢰구간의 그래프를 확인했을 때 최대값에서 예측값보다는 수치가 오를 수 있지만 평균적으로 확인했을 때 하락장을 나타내는 것을 확인할 수 있습니다.

### 5. 예측 결과 분석 및 설명

-   금 ETF과 리튬 ETF를 비교한 결과, 리튬 ETF에서 특정 시점을 기점으로 큰 변동(shock)이 발생한 것을 확인할 수 있었습니다
-   2018년 이전에는 리튬 ETF가 안정적인 모습을 보였으나, 2018년 이후 전기차 생산량이 급증하면서 해당 ETF 종목의 가격이 크게 상승했습니다.  
    이후 시간이 지남에 따라 다시 안정화되는 경향을 보이고 있습니다.

<img width="692" alt="스크린샷 2024-06-17 오후 11 33 03" src="https://github.com/kimgusan/Time_series/assets/156397911/5e340d17-ff14-46f2-bfcb-4220fa8571a2">

<img width="963" alt="스크린샷 2024-06-17 오후 11 30 54" src="https://github.com/kimgusan/Time_series/assets/156397911/71fa9adf-4606-4a18-b379-6368d73998f6">  

출처: https://www.weforum.org/agenda/2021/01/electric-vehicles-breakthrough-tesla-china

-   ARIMA 모델을 사용하여 2013년부터 2024년까지의 데이터를 평가한 결과, 예측 오차가 0.11 정도로 나타났으며 모델 평가가 좋았다고 판단됩니다.
-   Prophet 모델을 사용하여 2025년 리튬 ETF를 예측한 결과, 시간이 지날수록 ETF 가격이 하락하는 경향을 보였습니다.  
    이는 2020년에서 2023년 사이에 발생한 변동이 shock라는 부분에 신뢰성을 높여주는 결과입니다.  
-   해당 ETF는 비정상 시계열 데이터로 예측이 어려운 추세를 보였으며, 금과 같은 다중 공산성이 없는 자산과는 연관성이 없었습니다. 그러나 대용량 배터리를 만드는 데 사용되는 전기차 생산량과는 연관성이 있을 것으로 판단되었습니다.
-   신뢰구간의 그래프를 확인했을 때 최대값에서 예측값보다는 수치가 오를 수 있지만 평균적으로 확인했을 때 하락장을 나타내는 것을 확인할 수 있습니다.
-   위와 같은 결론으로 리튬 ETF의 경우 2020년 부터 발생한 주가 상승은 전기차의 급격한 수요에 의한 연관이 있을 것으로 예상되며 아직까지 전기차 수요가 다수를 차지 하지 않기 때문에 해당 주식에 대해서는 섣부른 투자는 하지 않는 것이 좋다고 사료됩니다.

### 6. 느낀점

-   시계열 데이터 분석에서 누적값을 사용하는지, 곱한 값을 사용하는지에 따라 시간이 지남에 따른 결과가 달라질 수 있다는 것을 확인했습니다.
-   모델 훈련에 사용되는 데이터의 날짜에 따라 예측 결과가 달라질 수 있으며, 너무 연관성이 없는 과거 데이터를 사용하기보다는 연관성이 있는 데이터를 사용하는 것이 시계열 데이터 훈련에 더 효과적이라는 점을 깨달았습니다.
