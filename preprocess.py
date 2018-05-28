import numpy as np 
import pandas as pd 
import os
import json
import codecs

train_nonr_fn = "../tweets/train_nonrumor.txt"
train_rumor_fn = "../tweets/train_rumor.txt"
test_nonr_fn = "../tweets/test_nonrumor.txt"
test_rumor_fn = "../tweets/test_rumor.txt"

with open(test_nonr_fn,'r') as f:
	s = ''; i = 3000
	for line in f:
		s += line; i+=1
		if i%3==0: # 3行一条微博
			data = {}
			ss = s.split('\n')
			socialFeature = ss[0].split('|')
			assert len(socialFeature) == 15
			data["tweetId"] = socialFeature[0]; data["userName"] = socialFeature[1]
			data["tweetUrl"] = socialFeature[2]; data["userUrl"] = socialFeature[3]
			data["publishTime"] = socialFeature[4]; data["original"] = socialFeature[5]
			data["retweetCount"] = socialFeature[6]; data["commentCount"] = socialFeature[7]
			data["praiseCount"] = socialFeature[8]; data["userID"] = socialFeature[9]
			data["AuthenticationType"] = socialFeature[10];	data["userFanCount"] = socialFeature[11]
			data["userFollowCount"] = socialFeature[12]; data["userTweetCount"] = socialFeature[13]
			data["publishPlatform"] = socialFeature[14]

			data["imagesID"]=[]
			imagesUrls = ss[1].split("|")[:-1]
			imagesUrls = [url.split("/")[-1] for url in imagesUrls]
			for picn in imagesUrls:
				pic_path = '../nonrumor_images/' + picn
				if os.path.isfile(pic_path) == True:
					data["imagesID"].append(picn)

			data["content"] = ss[2]
			data["class"] = 0 # 0 for nonrumor and 1 for rumor
			datajson = json.dumps(data, ensure_ascii=False)
			fwn = '../testData/test_' + str(int(i/3)) + '.json'
			with codecs.open(fwn, 'w', 'utf-8') as fw:
				fw.write(datajson)
			s = ''
			