## First about the environment

I'm using python 3.6 here for this homework (well python 3.6.13 exactly)

other packages can be installed by using pip $\rightarrow$ `pip install -r requirements.txt` 

執行方式：

```shell
python trader.py --training training.csv --testing testing.csv --output output.csv
```



## About data and the model

本次使用的data為TA提供的training.csv，此外無其他data。

關於訓練，最開始是用的上次作業一就有用過的LSTM模型，但是做下來，效果並不是很好，並且訓練時間太久了，於是，有找網路上關於回歸的一些分析文章，並問了幾位學長做回歸的經驗，決定用普通的機器學習模型，放棄深度學習。

這次選擇是用兩個模型，一個模型預測下一天的價格，另一個模型預測之後四天的平均價格，用這兩個價格決定實際的action：如果下一天的價格大於之後幾天的平均價，則賣出，因為代表之後幾天的趨勢是跌；如果下一天的價格小於之後幾天的平均價，則買入，因為代表之後幾天的趨勢是漲；除此之外都選擇無動作。

關於資料的處理，就是在讀入資料後，用sclearn.preprocessing裡的PolynomialFeatures，將open, high, low, close以及過去五天內的最高、最低、平均價格展開成36組feature，期待能得到更好的回歸效果。

關於回歸模型的選擇，有嘗試 ridge regression, lasso regression, support vector regression與random forest，最後選擇了表現相對更好的random forest。

現在已將 line 133 ~ line 138 註解掉，這部分是訓練以及保存模型的code，助教若要訓練模型，可以將這部分的註解打開。line 140 ~ line 141 是讀取之前有存好的模型，若不需要可以將其註解掉。

最後，用一個Trader進行預測結果的對比以及選擇合適的action。

