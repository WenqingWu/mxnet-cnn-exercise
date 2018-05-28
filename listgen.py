
import os
import json
import codecs


def make_list_data(set_name):

    # FORMAT ::= int_image_index \t label_index \t path_to_image \n

    #train
    fout = open(os.path.join('../list/', set_name+'_train.lst'), 'w')               #train list file

    for i in range(1, 7351):
        json_path = '../trainData/' + 'train_' + str(i) +'.json'
    
        with codecs.open(json_path, 'r', 'utf-8') as f:
            data = json.loads(f.read())
            
            index = int(data["tweetId"])                                            #int_image_index
            label = data["class"]                                                   #label_index 
            filename = ""                                                           #path_to_image

            if ((len(data["imagesID"])) and (data["imagesID"][0].find(".gif") < 0)): #does have jpg pictures
                filename = data["imagesID"][0]
            else:
                filename = ("3bb06d6djw1elxtxmebiaj20dw09ddh6.jpg" if (label > 0) else "4aa97819gw1evwkdt9t6wj20go0bn0v6.jpg")

            fout.write('%d\t%d\t%s\n'%(index, label, filename))

    fout.close()

    # test
    fout = open(os.path.join('../list/', set_name + '_test.lst'), 'w')              #test list file

    for i in range(1, 1997):
        json_path = '../testData/'+'test_' + str(i) + '.json'

        with codecs.open(json_path, 'r', 'utf-8') as f:
            data = json.loads(f.read())

            index = int(data["tweetId"])
            label = data["class"]
            filename = ""

            if ((len(data["imagesID"])) and (data["imagesID"][0].find(".gif") < 0)):
                filename = data["imagesID"][0]
            else:
                filename = ("3bb06d6djw1elxtxmebiaj20dw09ddh6.jpg" if (label > 0) else "4aa97819gw1evwkdt9t6wj20go0bn0v6.jpg")

            fout.write('%d\t%d\t%s\n'%(index, label, filename))
            
    fout.close()

make_list_data("rumor_vs_nonrumor")