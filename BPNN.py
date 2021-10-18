import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder,OrdinalEncoder 
from sklearn.model_selection import train_test_split

def load_data():
    dataset = pd.read_csv("credit_card_approval_classification.csv")
    print(dataset.isnull().sum().sort_values(ascending=False))
    feature = dataset[
            ['FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN','AMT_INCOME_TOTAL',
            'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE'
            ,'DAYS_BIRTH','DAYS_EMPLOYED','FLAG_PHONE','JOB']]
    target = dataset[["TARGET"]]
    return feature,target

input_dataset,output_dataset= load_data()

input_dataset = OrdinalEncoder().fit_transform(input_dataset)

input_dataset = MinMaxScaler().fit_transform(input_dataset)

oneHot = OneHotEncoder()
output_dataset = OneHotEncoder(sparse=False).fit_transform(output_dataset)

print("===============================")

from sklearn.decomposition import PCA
#PCA
pca= PCA(n_components=3)
component = pca.fit_transform(input_dataset)



#Split data train to 70% and data testing to 20%
x_train,x_test,y_train,y_test = train_test_split(component,output_dataset,test_size = 0.2,train_size=0.7)

#layers
import tensorflow as tf
layers = {
    "input":3,
    "hidden":3,#2/3 3 + 2 = 3
    "output":2,

}
input_hidden = {
    "weight":tf.Variable(tf.random.normal([layers["input"],layers["hidden"]])),
    "bias":tf.Variable(tf.random.normal([layers["hidden"]]))
}
hidden_output = {
    "weight":tf.Variable(tf.random.normal([layers["hidden"],layers["output"]])),
    "bias":tf.Variable(tf.random.normal([layers["output"]]))
}

#activate function using sigmoid 0 1
def activate(y):
    return tf.nn.sigmoid(y)


#feed forward
def feed_forward(input_dataset):
    #input ke hidden
    Wx1b = tf.matmul(input_dataset,input_hidden["weight"])+input_hidden["bias"]
    y1 = activate(Wx1b)
    #hidden ke output
    Wx2b= tf.matmul(y1,hidden_output["weight"])+hidden_output["bias"]
    y2 = activate( Wx2b)
    return y2

input_placeholder = tf.placeholder( tf.float32 ,[None, layers["input"]])

output_placeholder = tf.placeholder( tf.float32 ,[None, layers["output"]])

y_predict = feed_forward(input_placeholder)
#MSE
err = tf.reduce_mean(0.5 *(output_placeholder - y_predict)**2)

#gradient descent
learning_rate = 0.2
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(err)
epoch= 2000


#Saving model
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    k = 0
    loss_val = 0
    for i in range(1,epoch+1):
        
        train_dict = {
            input_placeholder : x_train,
            output_placeholder : y_train
        }
        sess.run(train, feed_dict=train_dict)
        loss = sess.run(err, feed_dict=train_dict)
        
        if i % 100 ==0:
            print(f"Epoch ke : {i}, Loss(Current Error/MSE) : {loss}")
        if i % 500==0:
            #Save untuk pertama kali
            if k == 0:
                saver.save(sess, 'model.ckpt')
                loss_val = loss
                k=1
            else:
                #Jika loss lebih kecil dari sebelumnnya maka disave
                if loss_val > loss:
                    loss_val = loss
                    saver.save(sess,'model.ckpt')
            
            print("Loss Vall:",loss_val,"Epoch ke: ",i)

with tf.Session() as sess:
    saver.restore(sess,'model.ckpt')
    test_dict = {
        # Test dataset 10%
        input_placeholder : x_test/10,
        output_placeholder : y_test/10 
    } 
       
    akurasi = tf.equal(tf.argmax(output_placeholder, axis=1), tf.argmax(y_predict, axis=1))
    result = tf.reduce_mean(tf.cast(akurasi, tf.float32))
    print(f'Akurasi : {sess.run(result, feed_dict=test_dict) * 100} %')
