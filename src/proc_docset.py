from xml.dom.minidom import parse
import xml.dom.minidom
import pdb
import os
import xml.etree.ElementTree as ET
from lxml import etree
import nltk
from nltk.tokenize import word_tokenize


'''
Official Documents for xml.dom:
https://docs.python.org/3/library/xml.dom.html#dom-nodelist-objects

An demo of xml.dom (in Chinese):
https://www.runoob.com/python/python-xml.html

Official Documents for xml.etree.ElementTree:
https://docs.python.org/3/library/xml.etree.elementtree.html?highlight=etree
'''

def tokenization(doc_id):
   '''
   We will first find the article based on the doc id. Articles published in 1996-2000 is in /corpora/LDC/LDC02T31/.
   Articles published in 2004-2006 is in /corpora/LDC/LDC08T25/data/
   '''
   result = '' #result contains headline of an article and its tokenization result
   dir = ''
   doc_id_length = len(doc_id)
   if (doc_id_length == 16):    
      dir = '/corpora/LDC/LDC02T31/'
   elif (doc_id_length == 21):
      dir = '/corpora/LDC/LDC08T25/data/'
   else:
      print("error, not in any dataset")
   # year = int(doc_id[8:12]) #e.g. '2004'
   # if (1996 <= year <= 2000):
   #    dir = '/corpora/LDC/LDC02T31/'
   # elif (2004 <= year <= 2006):
   #    dir = '/corpora/LDC/LDC08T25/data/'
   # else:
   #    print("error, not in any dataset")
   #    pdb.set_trace()
   
   subdir = ""
   if (doc_id_length == 21):
      docType = doc_id[0:7].lower() #e.g. 'xin_eng'
      dir += docType 
      articles = doc_id[0:14].lower() #e.g. 'xin_eng_200411'
   else:  # "APW19980719.0060"
      docType = doc_id[0:3].lower() #e.g. 'APW'
      year = doc_id[3:7]
      dir += docType + '/' + year
      articles = doc_id[3:11] + '_' + docType
      if (docType == 'APW' or 'XIE'):
         articles += '_' + 'ENG'


   tree = ET.parse(dir + '/' + articles + '.xml')
   if (doc_id_length == 21): #2004-2006
      root = tree.getroot()
      for child in range(len(root)):
         #print(root[child].tag, root[child].attrib)
         if (root[child].attrib['id'] == doc_id):
            headline = root[child][0].text.strip('\n')
            result += headline + '\n\n'
            format = False
            try:
               TEXT = root[child][2]
               if (TEXT[0].tag == 'P'):
                  format = True
               else:
                  print("strange format")
                  pdb.set_trace()         
            except:
               wholeTEXT = root[child][1]
               TEXT = wholeTEXT.text.split('\n\n') 

            if (format == True):
               for paragraph in TEXT:
                  para = paragraph.text.strip('\n')
                  sent_text = nltk.sent_tokenize(para) #split a paragraph to one or multiple sentences.
                  for sentence in sent_text:
                     tokens = word_tokenize(sentence) #For each sentence, tokenize it.           
                     for token in tokens:
                        result += token + ' '
                     result += '\n'
                  result += '\n'
            else:
               for paragraph in TEXT:
                  para = paragraph.strip('\n')
                  sent_text = nltk.sent_tokenize(para) #split a paragraph to one or multiple sentences.
                  for sentence in sent_text:
                     tokens = word_tokenize(sentence) #For each sentence, tokenize it.           
                     for token in tokens:
                        result += token + ' '
                     result += '\n'
                  result += '\n'

   else: #1996-2000
      # need to implement
      pass

   return result


def process(path):
   # read the .xml file using minidom parser
   mode = path.split('/')[6]
   DOMTree = xml.dom.minidom.parse(path)
   collection = DOMTree.documentElement
   topics = collection.getElementsByTagName("topic")

   for topic in topics:
      # print ("*****Topic*****")
      # topic_id = topic.getAttribute("id")
      # title = topic.getElementsByTagName("title")[0]
      # print ("title: %s" % title.childNodes[0].data)
      # narrative = topic.getElementsByTagName("narrative")[0]
      # print ("narrative: %s" % narrative.childNodes[0].data)
      docsetAs = topic.getElementsByTagName("docsetA")

      for docSetA in docsetAs:
         # For each docSetA, we will create a directory under output_dir. The name of the subdirectory is the same as the docsetA id.
         directory = docSetA.getAttribute("id")
         path = '../outputs/' + mode + '/' + directory
         if not os.path.exists(path):
            os.makedirs(path)
         doc = docSetA.getElementsByTagName("doc")
         length = doc.length
         for i in range(length):
            # print ("doc: %s" % doc.item(i).getAttribute("id"))
            doc_id = doc.item(i).getAttribute("id") 
            '''
            doc_id has two different formats. 
            1. "APW19990421.0284"   1996-2000
            2. "AFP_ENG_20061002.0523"   2004-2006
            '''
            # Open '../outputs/training/D0901A-AXIN_ENG_20041113.0001'
            output_file = open(path + '/' + doc_id, "w") 
            result = tokenization(doc_id)
            output_file.writelines(result)
            output_file.close()

def main():
   process("/dropbox/22-23/575x/Data/Documents/training/2009/UpdateSumm09_test_topics.xml")
   # process("/dropbox/22-23/575x/Data/Documents/evaltest/GuidedSumm11_test_topics.xml")
   # process("/dropbox/22-23/575x/Data/Documents/devtest/GuidedSumm10_test_topics")

if __name__ == "__main__":
    main()