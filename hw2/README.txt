因為logistic regression , best.py training需要花一些時間
所以shellscript執行這兩個model時都是先把weight寫死在裡面，
故在traincode資料夾裡放這兩個model原本的code，就是包含是怎麼train
data、計算準確率的code

traincode使用方法:
python3 best.py train.csv test.csv X_train Y_train X_test output.csv 
python3 logistic.py train.csv test.csv X_train Y_train X_test output.csv 
