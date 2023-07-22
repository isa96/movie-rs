# -*- coding: utf-8 -*-


!pip install opendatasets

import opendatasets as od
od.download("https://www.kaggle.com/aigamer/movie-lens-dataset")

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns

df_movie = pd.read_csv('/content/movie-lens-dataset/movies.csv')

"""# Data Understanding

Dataset ini didapat dari [kaggle](https://www.kaggle.com/). Dalam platform tersebut terdapat banyak dataset dari berbagai sumber dan perusahaan yang dapat membantu para pemula mengerti tentang dunia ilmuwan data. Untuk projek ini, saya mengambil data yang bernama [Movie Lens Dataset](https://www.kaggle.com/aigamer/movie-lens-dataset). Berikut adalah keterangan mengenai maksud dari variabel - variabel atau kolom tersebut :

- Ratings.csv
    - userId   : ID pengguna (data type : int 64)
    - movieId   : ID film     (data type : int 64)
    - rating    : Penilaian pengguna terhadap film terkait (data type : float 64)
    - timestamp : kode waktu film  (data type : int 64)
- Movie.csv
    - movieId   : ID film (data type : int 64)
    - title     : judul film bersama tahun rilis (data type : object)
    - genres    : Genre pada film tersebut (data type : object)
"""

df_movie

df_movie.info()

df_rating = pd.read_csv('/content/movie-lens-dataset/ratings.csv')

df_rating

df_rating.describe()

df_rating.info()

print("number of movies in rating : {}".format(len(df_rating.movieId.unique())))
print("Number of movies in movie list : {}".format(len(df_movie.	movieId.unique())))

"""# Data Understanding

Dalam proses data understanding, saya menggunakan visualisasi data berupa histogram karena saya ingin mengetahui seberapa banyak film yang dipublish dari dataset tersebut.Tentu pertama saya harus memisahkan atau membuat kolom baru untuk mengambil tahun dari kolom *title*.
"""

movie_analysis = df_movie.copy()
movie_analysis['year'] = df_movie['title'].str.extract('(\(\d+\))')
movie_analysis['year'] = movie_analysis['year'].str.extract('(\d+)').astype(float)
movie_analysis

movie_analysis.year.describe()

plt.figure(figsize=(20,8))
sns.histplot(data = movie_analysis,x='year',bins = 200)
plt.show()

"""Dari gambar tersebut, bisa dilihat bahwa tahun 1998 sampai 2005 yang memiliki film rilis terbanyak ketimbang tahun sebelumnya. """

genres=[]
for i in range(len(df_movie.genres)):
    for x in df_movie.genres[i].split('|'):
        if x not in genres:
            genres.append(x)  

len(genres)
for x in genres:
    df_movie[x] = 0
for i in range(len(df_movie.genres)):
    for x in df_movie.genres[i].split('|'):
        df_movie[x][i]=1
df_movie

"""Selanjutnya saya melihat jumlah genres dalam data tersebut. Saya ingin melihat seberapa banyak genre yang ada dalam dataset ini. Untuk mendapatkan jawaban dari kalimat sebelumnya, saya akan menggunakan visualisasi data yang sama seperti sebelumnya yaitu *bar plot*. """

view = df_movie.iloc[:,3:].sum().reset_index()
view.columns =['title','total']

genre = ['Adventure', 'Animation', 'Children',
       'Comedy', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller',
       'Horror', 'Mystery', 'Sci-Fi', 'War', 'Musical', 'Documentary', 'IMAX',
       'Western', 'Film-Noir', '(no genres listed)']

plt.figure(figsize=(12,6))
sns.barplot(x='title',y='total',data=view)
plt.xticks(rotation=45)
plt.plot

"""Gambar ini menunjukan bahwa genre *drama* adalah genre yang paling banyak jika dibandingkan dengan genre lainnya."""

df = df_rating.merge(df_movie,how='inner',on='movieId')

df.head()

df = df.drop(['timestamp','genres','Adventure', 'Animation', 'Children',
       'Comedy', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller',
       'Horror', 'Mystery', 'Sci-Fi', 'War', 'Musical', 'Documentary', 'IMAX',
       'Western', 'Film-Noir', '(no genres listed)'],axis=1)

df.head()

"""Selain itu, saya juga membuat *count plot* dari *library seaborn* untuk melihat 10 besar film yang sudah dirilis. *Count plot* sendiri adalah sebuah visualisasi data dari *seaborn* yang digunakan untuk menghitung seberapa banyak data dalam suatu label. """

movie_ratings = df.groupby('title')['rating'].count().reset_index().rename(columns={'rating':'total_rating'})

movie_ratings

visual_ratings = movie_ratings.sort_values(by ='total_rating',ascending=False)
plt.figure(figsize=(20,8))
sns.barplot(x='title',y='total_rating',data=visual_ratings.iloc[:10,:])
plt.xticks(rotation=45)
plt.plot

"""Jika kita lihat baik - baik, *Forrest Gump* memiliki tingkat popularitas yang tinggi karena film tersebut memiliki pemberian nilai terbanyak oleh para pengguna.

# Data Preparation

Dalam data preparation, ada beberapa teknik yang saya gunakan untuk proses *preparation*. Selain itu, ada 3 dataset yang saya akan periksa yaitu rating.csv yang dinamakan sebagai df_rating, Movie.csv yand dinamakan sebagai df_movie, dan gabungan kedua dataset yang dinamakan df. Berikut penjelasan beberapa teknik yang akan digunakan untuk *data preparation*dan hasil dari teknik tersebut :

1. Cek data null
    Data null dapat membuat suatu hasil prediksi model menjadi tidak akurat. Cara untuk melihat apakah data ini mengandung null atau tidak adalah dengan menggunakan *method* dari *library* *pandas* yaitu *isnull()*.Tetapi jikalau ada maka kita akan menggunakan kode berikut untuk menghapus data null.
    
    *dataframe.dropna()*
    
     Kode ini berfungsi untuk menghapuskan data yang memiliki null values di dalam row setiap data.
    
3. Cek duplikat data
    Selain data null, duplikat data juga bisa membuat model menjadi tidak akurat. Untuk memastikan apakah data memiliki data duplikat, maka kita akan menggunakan *method* lainnya yang juga berasal dari *pandas* yaitu *duplicated*.
    
4. Data encoding
    Untuk data encoding, dataset yang akan digunakan hanya df atau gabungan dari kedua dataset sebelumnya karena data yang akan digunakan untuk model adalah dataset df ini. Untuk penggunaanya, saya membuat encoding atau menyandikan nilai unik dari kolom user_id. Lalu saya melakukan proses encoding angka ke user_id. Hal yang serupa saya lakukan kepada item_id. Kemudian saya memetakan hasil dari encoding tersebut ke dalam dataframe df.  
    
5. One-Hot Encoding
     Proses ini digunakan untuk *cosine similarity*. Pertama saya membuat one hot encoding pada genre karena setiap film mempunyai jumlah genre yang berbeda dan genre yang bervariasi. Saya membuat kolom baru untuk setiap nilai genre yang terdapat dalam kolom genres.

     Data preparation ini digunakan untuk proses selanjutnya yaitu mengubah data one hot encoding ini menjadi data matrix.
     
     
5. Matrix

    Proses ini digunakan untuk *cosine similarity*. Hal ini dilakukan setelah saya melakukan one hot encoding pada kolom genres.Setelah itu, saya mengubah data yang sudah dimasukan kedalam one-hot encoding menjadi data matrix *compressed sparse row* dengan bantuan *library* *scipy*.
"""

df_rating.isnull().sum()

df_movie.isnull().sum()

df.isnull().sum()

check_duplicates = df_rating[df_rating.duplicated()]
print(check_duplicates)

check_duplicates = df_movie[df_movie.duplicated()]
print(check_duplicates)

check_duplicates = df[df.duplicated()]
print(check_duplicates)

"""# Cosine Similarity

## Data Preparation for Cosine Similarity
"""

df_movie_content = df_movie.drop(['movieId','genres'],axis=1)

df_movie_content = df_movie_content.set_index('title')

from scipy.sparse import csr_matrix
df_content = csr_matrix(df_movie_content.values)
type(df_content)

print(df_content)

"""# Deep Learning

## Data Preparation for Deep Learning
"""

# Mengubah userID menjadi list tanpa nilai yang sama
user_ids = df['userId'].unique().tolist()
print('list user_id: ', user_ids)
 
# Melakukan encoding userID
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
print('encoded userID : ', user_to_user_encoded)
 
# Melakukan proses encoding angka ke ke userID
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
print('encoded angka ke userID: ', user_encoded_to_user)

movie_ids = df['movieId'].unique().tolist()
 
# Melakukan proses encoding placeID
movie_to_movie_encoded = {x: i for i, x in enumerate(movie_ids)}
 
# Melakukan proses encoding angka ke placeID
movie_encoded_to_movie = {i: x for i, x in enumerate(movie_ids)}
 

 
# Mapping userID ke dataframe user
df['user'] = df['userId'].map(user_to_user_encoded)
 

df['movie'] = df['movieId'].map(movie_to_movie_encoded)

df

# Mendapatkan jumlah user
num_users = len(user_to_user_encoded)
print(num_users)
 
# Mendapatkan jumlah movie
num_movie = len(movie_encoded_to_movie)
print(num_movie)
 
# Mengubah rating menjadi nilai float
df['rating'] = df['rating'].values.astype(np.float32)
 
# Nilai minimum rating
min_rating = min(df['rating'])
 
# Nilai maksimal rating
max_rating = max(df['rating'])
 
print('Number of User: {}, Number of movie: {}, Min Rating: {}, Max Rating: {}'.format(
    num_users, num_movie, min_rating, max_rating
))

df = df.sample(frac=1, random_state=42)
df

# Membuat variabel x untuk mencocokkan data user dan movie menjadi satu value
x = df[['user', 'movie']].values
 
# Membuat variabel y untuk membuat rating dari hasil 
y = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
 
# Membagi menjadi 80% data train dan 20% data validasi
train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)

"""# Modeling and Result

Untuk tahap modeling, saya menggunakan *neural network* dan *Cosine Simirality* untuk sistem rekomendasi berbasis *collaborative filtering* dan *content-based filtering*. Untuk model deep learning, saya gunakan untuk sistem rekomendasi berbasis *collaborative filtering* dimana model ini akan menghasilkan rekomendasi untuk satu pengguna.

Untuk sistem rekomendasi berbasis *content-based filtering*, saya menggunakan *cosine similarity* yang akan menghitung kemiripan antara satu film dengan lainnya bedasarkan fitur yang terdapat pada satu film.Hasil dari model ini adalah pemberian 50 film rekomendasi bedasarkan genre.

Untuk model ini, saya sengaja memberi 50 rekomendasi film karena saya ingin menunjukan bahwa hasil dari model ini akan memberikan rekomendasi film dengan genre yang serupa dengan film yang direkomendasikan.

    
Untuk merangkum semua penjelasan, kedua model ini bisa digunakan untuk sistem rekomendasi berbasis *collaborative filtering* dan *content-based filtering*.

##Modelling for Cosine Similarity
"""

from sklearn.metrics.pairwise import cosine_similarity
 
# Menghitung cosine similarity pada matrix tf-idf
cosine_sim = cosine_similarity(df_content) 
cosine_sim

cosine_sim_df = pd.DataFrame(cosine_sim, index=df_movie['title'], columns=df_movie['title'])
print('Shape:', cosine_sim_df.shape)
 
# Melihat similarity matrix pada setiap resto
cosine_sim_df.sample(5, axis=1).sample(10, axis=0)

"""## Model Prediction for Cosine Similarity"""

def resto_recommendations(nama_resto, similarity_data=cosine_sim_df, items=df_movie[['title', 'genres','Adventure', 'Animation', 'Children',
       'Comedy', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller',
       'Horror', 'Mystery', 'Sci-Fi', 'War', 'Musical', 'Documentary', 'IMAX',
       'Western', 'Film-Noir', '(no genres listed)']], k=50):
    
 
    # Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan    
    # Dataframe diubah menjadi numpy
    # Range(start, stop, step)
    index = similarity_data.loc[:,nama_resto].to_numpy().argpartition(
        range(-1, -k, -1))
    
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    
    closest = closest.drop(nama_resto, errors='ignore')
 
    return pd.DataFrame(closest).merge(items).head(k)

movie = resto_recommendations('Retroactive (1997)')

movie

"""## Model Training for Deep Learning"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import  EarlyStopping
class RecommenderNet(tf.keras.Model):
 
  # Insialisasi fungsi
  def __init__(self, num_users, num_movie, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_movie = num_movie
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding( # layer embedding user
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(5e-7)
    )
    self.user_bias = layers.Embedding(num_users, 1) # layer embedding user bias
    self.movie_embedding = layers.Embedding( # layer embeddings movie
        num_movie,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(5e-7)
    )
    self.movie_bias = layers.Embedding(num_movie, 1) # layer embedding movie bias
 
  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0]) # memanggil layer embedding 1
    user_bias = self.user_bias(inputs[:, 0]) # memanggil layer embedding 2
    movie_vector = self.movie_embedding(inputs[:, 1]) # memanggil layer embedding 3
    movie_bias = self.movie_bias(inputs[:, 1]) # memanggil layer embedding 4
 
    dot_user_movie = tf.tensordot(user_vector, movie_vector, 2) 
 
    x = dot_user_movie + user_bias + movie_bias
    
    return tf.nn.sigmoid(x) # activation sigmoid

model = RecommenderNet(num_users, num_movie, 50) # inisialisasi model
 
# model compile
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    # metrics=[[tf.keras.metrics.Precision(),tf.keras.metrics.Recall()]]
    metrics=[[tf.keras.metrics.MeanAbsoluteError(),tf.keras.metrics.RootMeanSquaredError()]]
)
callbacks = EarlyStopping(
    min_delta=0.0001,
    patience=7,
    restore_best_weights=True,
)

history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 8,
    epochs = 100,
    validation_data = (x_val, y_val),
    callbacks=[callbacks]
)

"""## Model Prediction for Deep Learning"""

ratings = model.predict(user_movie_array).flatten()
 
top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_movie_ids = [
    movie_encoded_to_movie.get(movie_not_visited[x][0]) for x in top_ratings_indices
]
 
print('Showing recommendations for users: {}'.format(user_ID))
print('===' * 9)
print('movie with high ratings from user')
print('----' * 8)
 
top_movie_user = (
    movie_visited_by_user.sort_values(
        by = 'rating',
        ascending=False
    )
    .head(5)
    .movieId.values
)
 
movie_df_rows = movie_df[movie_df['movieId'].isin(top_movie_user)]
for row in movie_df_rows.itertuples():
    print(row.title)
 
print('----' * 8)
print('Top 10 movie recommendation')
print('----' * 8)
 
recommended_movie = movie_df[movie_df['movieId'].isin(recommended_movie_ids)]
for row in recommended_movie.itertuples():
    print(row.title)

"""# Evaluation 

Pada evaluation saya menggunakan tiga teknik yaitu *mean absolute error* , *root mean squared error*, dan metrik buatan saya yaitu *accurate*. bedasarkan [sumber](https://towardsdatascience.com/recommendation-systems-models-and-evaluation-84944a84fb8e) terkait, kedua metrik ini berhubungan dengan rating pengguna. Berikut adalah penjelasan terkait ketiga metrik ini :

- ***Mean Absolute Error*** : metrik ini digunakan untuk mengetahui kesalahan model atau memberitahu seberapa error model yang sudah di latih kepada data yang akan dites. berikut adalah rumus dari metrik tersebut.
        
    ![image](https://user-images.githubusercontent.com/82896196/135978354-10610b16-1ffd-4b38-aebc-04a8511baf0b.png)
    
    Dari sini, semakin rendahnya nilai MAE (*mean absolute error*) maka semakin baik dan akurat model yang dibuat.

- ***Root Mean Square Error*** : metrik ini juga menghitung seberapa error yang terdapat dari model. Semakin rendahnya nilai *root mean square error* semakin baik model tersebut dalam melakukan prediksi. dibawah ini adalah gambar dari formula *root mean square error*.

    ![image](https://user-images.githubusercontent.com/82896196/135995423-74268008-5509-4f61-8d16-df0372eb827e.png)
    
- ***accurate*** : untuk metrik ini, saya menghitung dengan total prediksi rekomendasi bedasarkan genre yang benar dibagi dengan total rekomendasi yang telah diberikan. Saya menggunakan metrik ini karena saya ingin mengetahui apakah model yang dipakai untuk *content-based learning* dapat memprediksi semua konten bedasarkan genre dengan benar. 
    
    
 Penggunaan kedua metrik tersebut bisa didapat dari model deep learning yang didapat saat melakukan model fitting pada data.  Dari hasil model tersebut, *mean absolute error* model ini adalah sebesar 0.1391 pada training dan 0.1516 pada test, sedangkan untuk *root mean squared error* model ini adalah 0.1815 pada tranining dan 0.1986 pada test. Hal ini menunjukan bahwa model ini memiliki error dibawah 20% jika menggunakan *mean absolute error* dan dibawah 20% jika menggunakan *root mean squared error*. Meskipun memiliki error sebesar kalimat sebelumnya, model ini masih bisa digunakan untuk sistem rekomendasi.
 
 Selain itu, untuk metrik *accurate* digunakan untuk mengevaluasi *cosine similarity*. Dari evaluasi ini, kita mendapatkan bahwa *cosine similarity* berfungsi dengan sempurna untuk merekomendasikan film karena hasil dari pengambilan sample secara acak menghasilkan akurasi 100% yang artinya tidak ada kesalahan dalam menggunakan *cosine similarity*

## Model Evaluation for Deep Learning
"""

plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model_metrics')
plt.ylabel('mean_absolute_error')
plt.xlabel('epoch')
plt.legend(['mean_absolute_error', 'val_mean_absolute_error'], loc='upper left')
plt.show()

plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model_metrics')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['root_mean_squared_error', 'val_root_mean_squared_error'], loc='upper left')
plt.show()

movie_df = df_movie

 
# Mengambil sample user
user_ID = df.userId.sample(1).iloc[0]
movie_visited_by_user = df[df.userId == user_ID]
 
# Operator bitwise (~), bisa diketahui di sini https://docs.python.org/3/reference/expressions.html 
movie_not_visited = movie_df[~movie_df['movieId'].isin(movie_visited_by_user.movieId.values)]['movieId'] 
movie_not_visited = list(
    set(movie_not_visited)
    .intersection(set(movie_to_movie_encoded.keys()))
)

 
movie_not_visited = [[movie_to_movie_encoded.get(x)] for x in movie_not_visited]
user_encoder = user_to_user_encoded.get(user_ID)
user_movie_array = np.hstack(
    ([[user_encoder]] * len(movie_not_visited), movie_not_visited)
)

"""# Model Evaluation for Cosine Similarity"""

retro = df_movie[df_movie['title'] == 'Retroactive (1997)']
get_genre = [i for i in genre if retro[i].values == 1]
df_movie[df_movie['title'] == 'Retroactive (1997)']["Sci-Fi"].values[0]

def accurate (name ):
  retro = df_movie[df_movie['title'] == name]
  get_genre = [i for i in genre if retro[i].values == 1]
  sum = float(0)
  for j in get_genre :
    print("The accuracy of "+ j+" : " + str((movie[j].sum()/len(movie[j]))*100) + "%")

accurate('Retroactive (1997)')

