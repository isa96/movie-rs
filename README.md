# Sistem Rekomendasi 

## Project Overview

Seiring perkembangannya jaman, film sudah menjadi hal yang wajar bagi para pengguna. Ditambah lagi sekarang film sudah hadir dalam bentukan 
website dimana para pengguna bisa menonton film tanpa perlu pergi ke bioskop. Tetapi, karena banyak film yang beredar di dunia, 
tentu itu membuat kita ragu tentang film apa yang menarik. Tidak mungkin jikalau kita menanyakan orang satu - satu untuk menegerti  film tersebut. 
Oleh karena itu, diperlukannya sistem rekomendasi. Menurut penelitian yang berjudul ["Penerapan Metode Deep Learning pada Sistem Rekomendasi Film"](https://github.com/Alvin-Buana/Recommendation_system/blob/main/Document.pdf), 
sistem rekomendasi adalah suatu aplikasi yang digunakan untuk memberikan rekomendasi dalam membuat suatu keputusan yang diinginkan pengguna. 

Dengan adanya sistem rekomendasi, *user experience* tentu akan lebih baik karena para pengguna bisa mengerti film yang ingin ditonton lebih baik. 
Pada projek ini, saya akan membuat sistem rekomendasi film yang berguna untuk bisa merekomendasikan film yang bisa ditonton pengguna dengan baik. 
Data yang digunakan dalam projek ini adalah data film yang sudah terisi rating oleh beberapa user sehingga jumlah rating yang diberikan pada 
setiap film tidak rata. Contohnya adalah satu film bisa saja memiliki ratusan rating dari berbagai user dan satu film bisa saja mempunyai beberapa 
pemberian rating oleh pengguna.

## Business Understanding

Permasalahan inti dari projek ini adalah karena banyak film yang dirilis setiap tahun maka pengguna menjadi ragu untuk memilih film yang ingin ditonton. 
Oleh karena itu, diperlukannya sistem rekomendasi dimana sistem tersebut bisa memberi film yang tepat untuk user. 

### Goal
Tujuan dari projek ini adalah untuk meningkatakan *user experience* saat mencari film yang ingin ditonton.

### Solution
Karena dataset terkait hanya berisi tentang rating atau hasil penilaian pengguna dan genre film, maka solusi yang sangat tepat untuk masalah 
ini adalah dengan menggunakan *collaborative filtering* dan *content-based filtering*. ***Collaborative Filtering*** merupakan cara 
untuk memberi rekomendasi bedasarkan penilaian komunitas pengguna atau biasa disebut dengan rating. Sedangkan ***Content-Based Filtering*** 
merupakan cara untuk memberi rekomendasi bedasarkan genre atau fitur pada item yang disukai oleh pengguna. Contoh dari *content-based filtering* adalah 
apabila pengguna menyukai film horror maka sistem akan merekomendasi film yang bertema horror pada pengguna.

Model yang saya akan gunakan untuk mendukung *collaborative filtering* yaitu dengan deep learning sedangkan untuk *content-based filtering* 
saya akan menggunakan cosine similarity. Berikut adalah penjelasan dari kedua model.

- ***Deep Learning*** : *deep learning* mempunyai banyak implementasi dalam menjawab setiap masalah. Untuk projek ini model deep learning yang digunakan akan menggunakan layer embedding yang merupakan layer untuk mengubah sebuah data menjadi vector yang dapat digunakan untuk proses selanjutnya. Kemudian setelah menggunakan layer embedding, hasil dari vector tersebut akan dimasukan ke dalam operasi vector dimana hasil dari operasi tersebut akan digunakan untuk dijumlahkan dengan bias yang lain. Terakhir, hasil dari total penjumlahan itu akan dimasukan ke dalam *neural network* dengan fungsi aktivasi *sigmoid*. Untuk optimizer, saya menggunakan optimizer Adam dan menggunakan loss Binary Crossentropy. Terakhir, untuk metrik saya menggunakan *root mean square error* dan *mean absolute error*. untuk penjelasan kedua metrik ini akan dibahas di subab selanjutnya.


- ***Cosine Similarity*** : *Cosine Similarity* merupakan model yang menghitung similaritas antara satu item dengan lainnya sehingga bisa dinyatakan bahwa item satu dengan lainnya mirip atau serupa. Cara menghitung *cosine similarity* adalah sebagai gambar dibawah ini :

![image](https://user-images.githubusercontent.com/82896196/137344839-c770d89e-0109-4f91-9691-813d818d0b64.png)


## Data Understanding

Dataset ini didapat dari [kaggle](https://www.kaggle.com/). Dalam platform tersebut terdapat banyak dataset dari berbagai sumber dan 
perusahaan yang dapat membantu para pemula mengerti tentang dunia ilmuwan data. Untuk projek ini, saya mengambil data yang bernama [Movie Lens Dataset](https://www.kaggle.com/aigamer/movie-lens-dataset). 
Berikut adalah keterangan mengenai maksud dari variabel - variabel atau kolom tersebut :

- Ratings.csv
    - userId   : ID pengguna (data type : int 64)
    - movieId   : ID film     (data type : int 64)
    - rating    : Penilaian pengguna terhadap film terkait (data type : float 64)
    - timestamp : kode waktu film  (data type : int 64)
- Movie.csv
    - movieId   : ID film (data type : int 64)
    - title     : judul film bersama tahun rilis (data type : object)
    - genres    : Genre pada film tersebut (data type : object)

Dalam proses data understanding, saya menggunakan visualisasi data berupa histogram karena saya ingin mengetahui seberapa banyak film yang dipublish dari dataset tersebut.Tentu pertama saya harus memisahkan atau membuat kolom baru untuk mengambil tahun dari kolom *title*. Berikut adalah histogram yang ditampilkan dari dataset **Movie.csv** :

![image](https://user-images.githubusercontent.com/82896196/137433911-95740ee3-2bad-47a4-adc8-709aef5e1068.png)

Dari gambar tersebut, bisa dilihat bahwa tahun 1998 sampai 2005 yang memiliki film rilis terbanyak ketimbang tahun sebelumnya. 

Selain itu, saya juga membuat *count plot* dari *library seaborn* untuk melihat 10 besar film yang sudah dirilis. *Count plot* sendiri adalah sebuah visualisasi data dari *seaborn* yang digunakan untuk menghitung seberapa banyak data dalam suatu label. Visualisasi ini didapatkan dari berapa banyak pengguna yang memberi penilaian kepada suatu film. Berikut adalah hasil dari visualisasi tersebut : 

![image](https://user-images.githubusercontent.com/82896196/137433769-9a1fe2b6-4454-47fe-862c-e88360946db5.png)

Jika kita lihat baik - baik, *Forrest Gump* memiliki tingkat popularitas yang tinggi karena film tersebut memiliki pemberian nilai terbanyak oleh para pengguna.

Selanjutnya saya melihat jumlah genres dalam data tersebut. Saya ingin melihat seberapa banyak genre yang ada dalam dataset ini. Untuk mendapatkan jawaban dari kalimat sebelumnya, saya akan menggunakan visualisasi data yang sama seperti sebelumnya yaitu *bar plot*.  Berikut adalah hasil untuk mengetahui jumlah genre pada film :

![image](https://user-images.githubusercontent.com/82896196/137434068-35b181e7-527b-41b0-904b-32404a63473f.png)


Gambar ini menunjukan bahwa genre *drama* adalah genre yang paling banyak jika dibandingkan dengan genre lainnya.

Jika kita hubungkan satu data dengan lainnya:

![image](https://user-images.githubusercontent.com/82896196/137434270-b87d5121-030c-4cd8-a631-5e7592fa6351.png)

Film yang berjudul "*Forrest Gump*" merupakan film favorit karena telah mendapat jumlah rating terbanyak dan genre film ini adalah genre yang mengandung drama. Sehingga kita bisa simpulkan bahwa pada saat itu banyak pengguna yang suka dengan film drama.


## Data Preparation 

Dalam data preparation, ada beberapa teknik yang saya gunakan untuk proses *preparation*. Selain itu, ada 3 dataset yang saya akan periksa yaitu rating.csv 
yang dinamakan sebagai df_rating, Movie.csv yand dinamakan sebagai df_movie, dan gabungan kedua dataset yang dinamakan df. Berikut penjelasan beberapa 
teknik yang akan digunakan untuk *data preparation*dan hasil dari teknik tersebut :

1. Cek data null
    Data null dapat membuat suatu hasil prediksi model menjadi tidak akurat. Cara untuk melihat apakah data ini mengandung null atau tidak adalah dengan menggunakan *method* dari *library* *pandas* yaitu *isnull()*. Berikut adalah hasil dari cek data null oleh *pandas* :
    
    ![image](https://user-images.githubusercontent.com/82896196/137434853-6598fa72-32fd-422d-ae58-27bb911b5f37.png)

    
     Dari hasil ketiga gambar ini, kita bisa simpulkan bahwa data ini tidak memiliki null sehingga kita tidak perlu melakukan teknik penghapusan data null. tetapi jikalau ada maka kita akan menggunakan kode berikut untuk menghapus data null.
    
    *dataframe.dropna()*
    
     Kode ini berfungsi untuk menghapuskan data yang memiliki null values di dalam row setiap data.
    
3. Cek duplikat data
    Selain data null, duplikat data juga bisa membuat model menjadi tidak akurat. Untuk memastikan apakah data memiliki data duplikat, maka kita akan menggunakan *method* lainnya yang juga berasal dari *pandas* yaitu *duplicated*. Berikut adalah hasil dari cek data duplikat :
    
    ![image](https://user-images.githubusercontent.com/82896196/137435012-3e7e50e1-0857-461b-829b-0fa7df89c2c3.png)

    Hasil dari gambar ini adalah tanda bahwa dataset kita merupakan dataset yang baik karena dataset ini tidak memiliki duplikat ataupun data null. 
    
4. Data encoding

    Untuk data encoding, dataset yang akan digunakan hanya df atau gabungan dari kedua dataset sebelumnya karena data yang akan digunakan untuk model adalah dataset df ini. Untuk penggunaanya, saya membuat encoding atau menyandikan nilai unik dari kolom user_id. Lalu saya melakukan proses encoding angka ke user_id. Hal yang serupa saya lakukan kepada item_id. Kemudian saya memetakan hasil dari encoding tersebut ke dalam dataframe df. Hasil dari data encoding adalah sebagai berikut :
    
    ![image](https://user-images.githubusercontent.com/82896196/137435207-03f1f1ae-ff48-478d-b442-a0186a325184.png)

    Hasil dari data encoding ini akan digunakan untuk model deep learning. 
    
5. One-Hot Encoding
     Proses ini digunakan untuk *cosine similarity*. Pertama saya membuat one hot encoding pada genre karena setiap film mempunyai jumlah genre yang berbeda dan genre yang bervariasi. Saya membuat kolom baru untuk setiap nilai genre yang terdapat dalam kolom genres. Berikut adalah hasil dari one-hot encoding tersebut : 
     
     ![image](https://user-images.githubusercontent.com/82896196/137435663-34d58040-1351-4332-bc03-0bdb3529febc.png)

     Data preparation ini digunakan untuk proses selanjutnya yaitu mengubah data one hot encoding ini menjadi data matrix.
     
     
6. Matrix

    Proses ini digunakan untuk *cosine similarity*. Hal ini dilakukan setelah saya melakukan one hot encoding pada kolom genres.Setelah itu, saya mengubah data yang sudah dimasukan kedalam one-hot encoding menjadi data matrix *compressed sparse row* dengan bantuan *library* *scipy*.  Berikut adalah hasil dari metode data preparation ini :
    
    ![image](https://user-images.githubusercontent.com/82896196/137435502-059a9eab-1c62-4dee-87fa-f7193b2a34a8.png)

    

Selanjutnya kita akan masuk ke dalam tahap modeling dan result.


## Modeling and Result

Untuk tahap modeling, saya menggunakan *neural network* dan *Cosine Simirality* untuk sistem rekomendasi berbasis *collaborative filtering* dan *content-based filtering*. Untuk model deep learning, saya gunakan untuk sistem rekomendasi berbasis *collaborative filtering* dimana model ini akan menghasilkan rekomendasi untuk satu pengguna. Berikut adalah hasil prediksi dari model deep learning tersebut :

   ![image](https://user-images.githubusercontent.com/82896196/137435919-d69f9353-5f1f-4796-bd25-be06cc345e20.png)
    
Pada hasil gambar ini, kita bisa melihat bahwa pengguna id nomor 226 sangat menyukai Willy Wonka & the Chocolate Factory (1971) , Office Space (1999) , Wayne's World (1992) ,Endless Summer, The (1966) ,dan Bill & Ted's Excellent Adventure (1989). Oleh sebab itu, sistem ini akan merekomendasikan 10 film yang mirip dengan hasil tersebut seperti Silence of the Lambs, The (1991). 

Untuk sistem rekomendasi berbasis *content-based filtering*, saya menggunakan *cosine similarity* yang akan menghitung kemiripan antara satu film dengan lainnya bedasarkan fitur yang terdapat pada satu film.Hasil dari model ini adalah pemberian 50 film rekomendasi bedasarkan genre. Berikut adalah hasil dari *cosine similarity* :

   ![image](https://user-images.githubusercontent.com/82896196/137436303-848a53f0-f9df-4634-91e0-e2b70baa2ecc.png)

Untuk model ini, saya sengaja memberi 50 rekomendasi film karena saya ingin menunjukan bahwa hasil dari model ini akan memberikan rekomendasi film dengan genre yang serupa dengan film yang direkomendasikan.

    
Untuk merangkum semua penjelasan, kedua model ini bisa digunakan untuk sistem rekomendasi berbasis *collaborative filtering* dan *content-based filtering*.

## Evaluation 

Pada evaluation saya menggunakan tiga teknik yaitu *mean absolute error* , *root mean squared error*, dan metrik buatan saya yaitu *precision*. bedasarkan [sumber](https://towardsdatascience.com/recommendation-systems-models-and-evaluation-84944a84fb8e) terkait, kedua metrik ini berhubungan dengan rating pengguna. Berikut adalah penjelasan terkait ketiga metrik ini :

- ***Mean Absolute Error*** : metrik ini digunakan untuk mengetahui kesalahan model atau memberitahu seberapa error model yang sudah di latih kepada data yang akan dites. berikut adalah rumus dari metrik tersebut.
        
    ![image](https://user-images.githubusercontent.com/82896196/135978354-10610b16-1ffd-4b38-aebc-04a8511baf0b.png)
    
    Dari sini, semakin rendahnya nilai MAE (*mean absolute error*) maka semakin baik dan akurat model yang dibuat.

- ***Root Mean Square Error*** : metrik ini juga menghitung seberapa error yang terdapat dari model. Semakin rendahnya nilai *root mean square error* semakin baik model tersebut dalam melakukan prediksi. dibawah ini adalah gambar dari formula *root mean square error*.

    ![image](https://user-images.githubusercontent.com/82896196/135995423-74268008-5509-4f61-8d16-df0372eb827e.png)
    
- ***precision*** : untuk metrik ini, saya menghitung dengan total prediksi rekomendasi bedasarkan genre yang benar dibagi dengan total rekomendasi yang telah diberikan. Saya menggunakan metrik ini karena saya ingin mengetahui apakah model yang dipakai untuk *content-based learning* dapat memprediksi semua konten bedasarkan genre dengan benar. 
    
    
 Penggunaan kedua metrik tersebut bisa didapat dari model deep learning yang didapat saat melakukan model fitting pada data.  Dari hasil model tersebut, *mean absolute error* model ini adalah sebesar 0.1391 pada training dan 0.1516 pada test, sedangkan untuk *root mean squared error* model ini adalah 0.1815 pada tranining dan 0.1986 pada test. Hal ini menunjukan bahwa model ini memiliki error dibawah 20% jika menggunakan *mean absolute error* dan dibawah 20% jika menggunakan *root mean squared error*. Meskipun memiliki error sebesar kalimat sebelumnya, model ini masih bisa digunakan untuk sistem rekomendasi.
 
 Selain itu, untuk metrik *precision* digunakan untuk mengevaluasi *cosine similarity*. Dari evaluasi ini, kita mendapatkan bahwa *cosine similarity* berfungsi dengan sempurna untuk merekomendasikan film karena hasil dari pengambilan sample secara acak menghasilkan akurasi 100% yang artinya tidak ada kesalahan dalam menggunakan *cosine similarity*
 
 
 ## Pernutup
 Dengan berakhirnya penjelasan metrik, berakhir juga laporan ini. Terima kasih karena telah membaca laporan ini. 
 Saya harap apa yang saya sudah sampaikan dapat menjadi bermanfaat bagi yang membaca laporan ini.
