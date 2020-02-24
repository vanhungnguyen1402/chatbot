# Import các thư viện cần thiết, đặc biệt là các thư viện được xử dụng trong NLP như NLTK, tensorflow...
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import random
from tkinter import*

tk = Tk()
tk.geometry("600x400")
#Loading data training. Vì dữ liệu traning của chúng ta đang ở dạng json nên import json vào.
import json
with open('intents.json') as json_data:
    intents = json.load(json_data) #Với intents.json vừa được đưa vào bộ training 
#Bạn nên sử dụng thêm bộ công cụ của xử lý ngôn ngữ tự nhiên nltk để tiền xử lý dữ liệu. 
# Bộ công cụ này cho phép bạn thực hiện các quá trình tokenizer, POS stagging, word segmentation, 
# remove stopword....
words = []
classes = []
documents = []
stop_words = ['?', 'a', 'an', 'the']

for intent in intents['intents']:
    for pattern in intent['patterns']:

        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])   

words = [stemmer.stem(w.lower()) for w in words if w not in stop_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))
#Bổ sung thêm stopwords vào stop_words array hoặc tạo một file text để list danh sách các stopwords
# stopwords là gì: là danh sách các từ xuất hiện nhiều trong văn bản nhưng lại không có giá trị trong việc phân lớp. 
# Vì vậy trước khi training, chúng ta cần làm sạch văn bản, loại bỏ những từ ngữ không có ý nghĩa này để 
# tránh overfiting ảnh hưởng đến kết quả training.

#Không may, nếu chúng ta sử dụng dự liệu dạng word như vậy thì sẽ không thể hoạt động được với tensorflow, 
# vì vậy việc cần thiết bây giờ là chuyển dữ liệu này sang dạng tensor number

# create our training data
training = []
output = []
output_empty = [0] * len(classes)

# training set, bag of words for each sentence(tập huấn luyện, túi từ cho mỗi câu)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
random.shuffle(training)
training = np.array(training)

# create train and test lists
train_x = list(training[:,0])
train_y = list(training[:,1])

#Lưu ý rằng, dữ liệu của chúng ta bị xáo trộn. Tensorflow sẽ lấy một số dữ liệu trong tập intents.json 
# để làm dữ liệu thử nghiệm để đo độ chính xác cho mô hình. Sau khi chuyển word -> number, 
# bạn có thể thấy các "bag-of-words" array dạng như sau:



# build neural network để training model
# chúng ta đang xây dựng mạng neural 2 lớp để traning
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')


#Để hoàn thành quá trình này, bạn cần lưu lại (pickle) model và document để có thể sử dụng lại nó 
# trong quá trình predict ở bước tiếp theo.
import pickle
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )

#Predict (Xây dựng ChatBot)
#xây dựng hệ thống phản hồi của Chatbot, sử dụng intents model đã được training ở bước trên. 
# Sau khi import các thư viện giống như bước training ở trên, Bạn cần un-pickle model và documents, 
# cũng như cần phải load lại intents.json (cái này để lấy các response đã được định nghĩa trước đó
# restore our data structures
import pickle
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import intents file
import json
with open('intents.json') as json_data:
    intents = json.load(json_data)

#Load tensorflow model
model.load('./model.tflearn')


#Cũng giống như khi training dữ liệu, với các ý định người dùng nhập vào bạn cũng cần phải tiền xử lý, tokenizer, hay chuyển sang bag-of-words để hệ thống có thể hiểu và phân loại về đúng lớp của nó
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# bag of words
def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

#Vậy là dữ liệu đầu vào đã xử lý xong. Và bây giờ là bộ xử lý phản hồi của chatbot
# data structure to hold user context
context = {}

ERROR_THRESHOLD = 0.25
def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list

def response(sentence, userID='1', show_details=False):
    results = classify(sentence)
    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        return random.choice(i['responses'])

            
            results.pop(0)

#Với mỗi câu hỏi (hay request) người dùng nhập vào, tôi sẽ sử dụng hàm model.predict() 
# để xác định request đó thuộc loại nào. Sau đó sẽ đưa ra các phản hồi tiềm ẩn, 
# có khả năng và phù hợp nhất với các request trước đó. Và kết quả ....
#Gọi hàm
#cauhoi = str(input("Nhap cau hoi: "))


def ghi():
    cauhoi = cau_hoi.get()
     #cauhoi =str(cauhoi)
    b = response(cauhoi)
    lable2.configure(text=b)



lable1 = Label(tk, text="You want ask me about:", fg="red")
lable1.place(x=10, y=10)
cau_hoi = Entry(tk, bd =5)
cau_hoi.place(x= 200, y=10)
lable2 = Label(tk, text=" ", fg="red")
lable2.place(x = 10, y = 90)
btn1 = Button(tk, text="Answer", command=ghi, fg='red')
btn1.place(x= 200, y =50)





tk.mainloop()




//=======================================================================================================================
def ghi():
    cauhoi = cau_hoi.get()
     #cauhoi =str(cauhoi)
    b = response(cauhoi)
    lable2.configure(text=b)def ghi():
    cauhoi = cau_hoi.get()
     #cauhoi =str(cauhoi)
    b = response(cauhoi)
    lable2.configure(text=b)def ghi():
    cauhoi = cau_hoi.get()
     #cauhoi =str(cauhoi)
    b = response(cauhoi)
    lable2.configure(text=b)def ghi():
    cauhoi = cau_hoi.get()
     #cauhoi =str(cauhoi)
    b = response(cauhoi)
    lable2.configure(text=b)def ghi():
    cauhoi = cau_hoi.get()
     #cauhoi =str(cauhoi)
    b = response(cauhoi)
    lable2.configure(text=b)def ghi():
    cauhoi = cau_hoi.get()
     #cauhoi =str(cauhoi)
    b = response(cauhoi)
    lable2.configure(text=b)def ghi():
    cauhoi = cau_hoi.get()
     #cauhoi =str(cauhoi)
    b = response(cauhoi)
    lable2.configure(text=b)
Mr.Nicholasking123