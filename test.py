import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.api as sm
df_movies = pd.read_csv('[DS]TermProject_data.csv', encoding='cp949')


df_actors = pd.read_csv('df_actor.csv')
# 영화별 배우 리스트 생성
actor_grouped = df_actors.groupby('영화명')['배우'].apply(list).reset_index()


# 영화 데이터와 배우 데이터 병합
df_movies = pd.merge(df_movies, actor_grouped, on='영화명', how='left')



# 배우 리스트를 문자열로 변환 
#df_movies['배우'] = df_movies['배우'].apply(lambda x: ','.join(x) if isinstance(x, list) else np.nan)
df_movies['배우'] = df_movies['배우'].apply(lambda x: x if isinstance(x, list) else [])

# 배우 데이터가 없는 영화 데이터 제거
df_movies = df_movies.dropna(subset=['배우'])


# 전국 매출액의 평균을 기준으로 흥행 여부(Y/N) 레이블 생성
mean_revenue = df_movies['전국 매출액'].mean()
df_movies['흥행'] = np.where(df_movies['전국 매출액'] > mean_revenue, 'Y', 'N')

# 데이터 확인
df_movies.info()

#다중 라벨 이진화(multilabelbinarizer)

# MultiLabelBinarizer를 사용하여 배우 더미 변수 생성
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
df_actors_dummies = pd.DataFrame(mlb.fit_transform(df_movies['배우']), columns=mlb.classes_, index=df_movies.index)

# 영화 데이터프레임과 배우 더미 변수 병합
df_movies = pd.concat([df_movies, df_actors_dummies], axis=1)
df_movies.drop('배우', axis=1, inplace=True)