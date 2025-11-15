# 시각화를 위한 라이브_러리들을 불러옵니다.
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # 이전 단계에서 사용했지만, 코드의 독립성을 위해 다시 포함합니다.

# --------------------------------------------------------------------------
# 0. 이전 단계: 데이터 불러오기 (경로는 사용자님 환경에 맞게 확인해주세요)
# --------------------------------------------------------------------------
file_path = r'C:\Users\shin\machineLearning\data\01_raw\WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(file_path)


# --------------------------------------------------------------------------
# 1. 데이터 정제 (Cleaning): TotalCharges 컬럼을 숫자로 바꾸기
# --------------------------------------------------------------------------
# TotalCharges 컬럼을 숫자로 변환합니다. 변환이 불가능한 값(예: 공백)은 NaN(결측치)으로 만듭니다.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# TotalCharges에 결측치가 있는지 확인합니다.
print(f"--- [TotalCharges 컬럼 결측치 개수] ---\n{df['TotalCharges'].isnull().sum()}개")

# 결측치가 있다면, 전체 데이터의 중간값(median)으로 채워넣습니다.
# 평균보다 중간값을 사용하는 것이 극단적인 값의 영향을 덜 받아 안정적입니다.
if df['TotalCharges'].isnull().sum() > 0:
    median_val = df['TotalCharges'].median()
    df['TotalCharges'].fillna(median_val, inplace=True)
    print("결측치를 중간값으로 채웠습니다.")


# --------------------------------------------------------------------------
# 2. 데이터 시각화: 고객 이탈 현황 파악하기
# --------------------------------------------------------------------------
# 그래프 스타일과 폰트 크기를 설정합니다.
sns.set(style="whitegrid")
plt.figure(figsize=(12, 18)) # 전체 그림의 크기를 설정합니다.

# (1) 고객 이탈 비율 (Churn)
plt.subplot(3, 2, 1) # 3행 2열의 첫 번째 위치에 그래프를 그립니다.
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution (Target)', fontsize=14)
plt.xlabel('Churn (Yes: 이탈, No: 유지)')
plt.ylabel('Number of Customers')


# (2) 계약 유형(Contract)에 따른 이탈률
plt.subplot(3, 2, 2)
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title('Churn by Contract Type', fontsize=14)
plt.xlabel('Contract Type')
plt.ylabel('Number of Customers')


# (3) 인터넷 서비스(InternetService)에 따른 이탈률
plt.subplot(3, 2, 3)
sns.countplot(x='InternetService', hue='Churn', data=df)
plt.title('Churn by Internet Service', fontsize=14)
plt.xlabel('Internet Service')
plt.ylabel('Number of Customers')


# (4) 요금 지불 방식(PaymentMethod)에 따른 이탈률
plt.subplot(3, 2, 4)
sns.countplot(x='PaymentMethod', hue='Churn', data=df, order = df['PaymentMethod'].value_counts().index)
plt.title('Churn by Payment Method', fontsize=14)
plt.xticks(rotation=15) # x축 라벨이 길어서 살짝 회전시킵니다.
plt.xlabel('Payment Method')
plt.ylabel('Number of Customers')


# (5) 월 요금(MonthlyCharges) 분포
plt.subplot(3, 2, 5)
sns.histplot(data=df, x='MonthlyCharges', hue='Churn', kde=True, multiple="stack")
plt.title('Churn by Monthly Charges', fontsize=14)
plt.xlabel('Monthly Charges')
plt.ylabel('Number of Customers')


# (6) 총 사용 기간(tenure) 분포
plt.subplot(3, 2, 6)
sns.histplot(data=df, x='tenure', hue='Churn', kde=True, multiple="stack")
plt.title('Churn by Tenure', fontsize=14)
plt.xlabel('Tenure (Months)')
plt.ylabel('Number of Customers')


# 그래프들이 서로 겹치지 않게 레이아웃을 조정하고 화면에 보여줍니다.
plt.tight_layout()
plt.show()