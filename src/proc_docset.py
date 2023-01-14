from xml.dom.minidom import parse
import xml.dom.minidom
import pdb
import os
import xml.etree.ElementTree as ET
from lxml import etree
import nltk
from nltk.tokenize import word_tokenize
import re


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
      articles = doc_id[3:11] + '_' + docType.upper()
      if (docType == 'apw' or docType == 'xie'):
         articles += '_' + 'ENG'

   corpora = '1' # 1:AQUAINT-2, 2:AQUAINT, 3:TAC 2011 shared task


   try: #AQUAINT-2
      tree = ET.parse(dir + '/' + articles)
   except:
      try:
         tree = ET.parse(dir + '/' + articles + '.xml')
      except:
         try: #AQUAINT
            corpora = '2'
            f = open(dir + '/' + articles, 'r')
            parser = etree.XMLParser(recover=True)
            tree = etree.parse(f, parser)
         except: #TAC 2011 shared task
            try:
               corpora = '3'
               dir = '/corpora/LDC/LDC10E12/12/TAC_2010_KBP_Source_Data/data/2009/nw/'
               subdir = docType[0:3] + '_' + doc_id[4:7].lower() + '/' + doc_id[8:16]
               allFile = os.listdir(dir+subdir)
               for file in allFile:
                  if (doc_id in file):
                     tree = ET.parse(dir+subdir+'/'+file)
            except:
               print("can't find xml rile")
               pdb.set_trace()


   if (corpora == '1'): #2004-2006
      root = tree.getroot()
      for child in range(len(root)):
         #print(root[child].tag, root[child].attrib)
         if (root[child].attrib['id'] == doc_id):
            headline = root[child][0].text.strip('\n')
            result += headline + '\n\n'
            format = False
            # if (doc_id == "APW_ENG_20041118.0081"):
            #    pdb.set_trace()
            try:
               TEXT = root[child][2]
               if (TEXT[0].tag == 'P'):
                  format = True
               else:
                  print("unexpected format1")
                  pdb.set_trace()         
            except:
               try:
                  if (root[child][1][0].tag == 'P'):
                     format = True
                     TEXT = root[child][1]
                  else:
                     print("unexpected format2")
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

   elif (corpora == '2'): #1996-2000  
      '''
      DOMTree = xml.dom.minidom.parse(dir + '/' + articles)
      collection = DOMTree.documentElement
      topics = collection.getElementsByTagName("topic")
      '''
      # Having trouble with this part cause the xml files have very different format

      # root = tree.getroot()
      # itr = tree.getiterator
      # DOCNO = tree.findall("DOCNO")
      # for child in range(len(root)):  
      #    pdb.set_trace()
      #    if (doc_id in root[child].text):
      #       if (root[child][2].tag = 'BODY'):
      #          headline = root[child][2][0]
      #    pdb.set_trace()
      # need to implement
      pass

   else:
      root = tree.getroot()

      for child in range(len(root)):
         if (root[child].tag == 'BODY'):
            for i in range(len(root[child])):
               if (root[child][i].tag == 'HEADLINE'):
                  headline = root[child][i].text.strip('\n')
                  result += headline + '\n\n'
               if (root[child][i].tag == 'TEXT'):
                  for j in range (len(root[child][i])):
                     para = root[child][i][j].text.strip('\n')
                     sent_text = nltk.sent_tokenize(para) #split a paragraph to one or multiple sentences.
                     for sentence in sent_text:
                        tokens = word_tokenize(sentence) #For each sentence, tokenize it.           
                        for token in tokens:
                           result += token + ' '
                        result += '\n'
                     result += '\n'        
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
   process("/dropbox/22-23/575x/Data/Documents/evaltest/GuidedSumm11_test_topics.xml")
   # process("/dropbox/22-23/575x/Data/Documents/devtest/GuidedSumm10_test_topics.xml")

if __name__ == "__main__":
    main()