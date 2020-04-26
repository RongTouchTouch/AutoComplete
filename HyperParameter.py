procedure="train"
chunk_len=25   #每次训练用多少个字符
A_end_index=20      #the end index of A in the chunk
B_start_index=5       #the start index of B in the chunk
batch = 16        #每批几个输入输出
nbatches = 16       #每轮训练几批
epoch_number=100001 #要训练多少轮
epoches_of_loss_record=100 #每多少轮将损失输出到文本
epoches_of_model_save=500   #每多少轮存储一次模型
transformer_size=6  #encoder和decoder有多少层
predict_length=8   #预测结果的长度（字符数）
beam_search_number= 3 #束搜索的大小
bleu_model_name='transformer_kg_1000_10_9_3.model'

