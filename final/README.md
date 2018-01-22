# ML final project ： TV Conversation
- **組名** : NTU_b04902092_CGSS
- 組員:
    1. b04902003 董舒博
    2. b04902082 鍾偉傑
    3. b04902092 張均銘
    4. r06921075 許耀文

## 使用套件
- **python3** == 3.5
- **gensim** == 3.2.0
- **jieba** == 0.39
- **numpy** == 1.13.3
- **sklearn** == 0.19.1
- **hanziconv** == 0.3.2

## 檔案介紹
- **src/**
    - **test.sh** : 執行程式的shellscript
    - **test.py** : load model並predict testing data的程式
    - **bestmodel.bin** : 目前產生的最佳model(kaggle public分數最高)
- **Report.pdf**: Report內容

## 使用方法
 `cd src`
 
 `./test.sh $1 $2`
> $1 : testing data path
> $2 : output csvfile path

## 流程
1. **安裝上述gensim,jieba,numpy,sklearn,hanziconv套件**
2. **進入src資料夾內執行shellscript test.sh**
3. **test.py載入bestmodel.bin**
4. **開始切testdata、算cosine similarity**
5. **輸出output.csv到 outputfile的path**
