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
   If an article is not in the two dirs above, then it is in /corpora/LDC/LDC10E12/12/TAC_2010_KBP_Source_Data/data/2009/nw/
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
      if (docType == 'apw' or docType == 'xie'):  # '19980601_APW_ENG' compare with '19980601_NYT'
         articles += '_' + 'ENG'
      if (docType == 'xie'):
         articles = articles.replace('XIE', 'XIN')


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
            lines = f.readlines()
            lines.insert(0, '<tag>\n')
            lines.append('</tag>\n')
            tmpFile = open('tmp.txt','w')
            for item in lines:
	            tmpFile.write(item)
            tmpFile.close()
            f = open('tmp.txt', 'r')
            parser = etree.XMLParser(recover=True)
            tree = etree.parse(f, parser)
            f.close()
            os.remove('tmp.txt')
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
            result += 'headline: ' + headline + '\n\n'
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
      root = tree.getroot()
      for child in range(len(root)): 
         if (doc_id in root[child][0].text):         
            if (root[child][2].tag == 'BODY'):
               if (root[child][2][1].tag == 'HEADLINE'):
                  headline = root[child][2][1].text
                  result += 'headline: ' + headline + '\n\n'
               elif (root[child][2][0].tag == 'HEADLINE'):
                  headline = root[child][2][1].text
                  result += 'headline: ' + headline + '\n\n'
               try:
                  if (root[child][2][2].tag == 'TEXT'):
                     TEXT = root[child][2][2]
                     for p in TEXT:
                        para = p.text.strip('\n')
                        sent_text = nltk.sent_tokenize(para) #Split a paragraph to one or multiple sentences.
                        for sentence in sent_text:
                           tokens = word_tokenize(sentence) #For each sentence, tokenize it.           
                           for token in tokens:
                              result += token + ' '
                           result += '\n'
                        result += '\n'
               except:
                  try:
                     if (root[child][2][1].tag == 'TEXT'):
                        TEXT = root[child][2][1]
                        for p in TEXT:
                           para = p.text.strip('\n')
                           sent_text = nltk.sent_tokenize(para) #Split a paragraph to one or multiple sentences.
                           for sentence in sent_text:
                              tokens = word_tokenize(sentence) #For each sentence, tokenize it.           
                              for token in tokens:
                                 result += token + ' '
                              result += '\n'
                           result += '\n'
                  except:
                     print("different format 1")
                     pdb.set_trace()
            elif (root[child][3].tag == 'BODY'):
               if (root[child][3][0].tag == 'HEADLINE'):
                  headline = root[child][3][0].text
                  result += 'headline: ' + headline + '\n\n'
  
               if (root[child][3][1].tag == 'TEXT'):
                  TEXT = root[child][3][1]
                  for p in TEXT:
                     para = p.text.strip('\n')
                     sent_text = nltk.sent_tokenize(para) #Split a paragraph to one or multiple sentences.
                     for sentence in sent_text:
                        tokens = word_tokenize(sentence) #For each sentence, tokenize it.           
                        for token in tokens:
                           result += token + ' '
                        result += '\n'
                     result += '\n'
  
               else:
                  print("different format 4")
                  pdb.set_trace()
            elif (root[child][4].tag == 'BODY'):
               # if (root[child][4][1] == 'HEADLINE'):
               #    headline = root[child][4][1].text
               if (root[child][4][0].tag == 'TEXT'):
                     # no headline
                     TEXT = root[child][4][0]
                     for p in TEXT:
                        para = p.text.strip('\n')
                        sent_text = nltk.sent_tokenize(para) #Split a paragraph to one or multiple sentences.
                        for sentence in sent_text:
                           tokens = word_tokenize(sentence) #For each sentence, tokenize it.           
                           for token in tokens:
                              result += token + ' '
                           result += '\n'
                        result += '\n'

               elif (root[child][4][1].tag == 'TEXT'):
                  # no headline
                  result += 'headline: ' + '\n\n'
                  TEXT = root[child][4][1]
                  for p in TEXT:
                     para = p.text.strip('\n')
                     sent_text = nltk.sent_tokenize(para) #Split a paragraph to one or multiple sentences.
                     for sentence in sent_text:
                        tokens = word_tokenize(sentence) #For each sentence, tokenize it.           
                        for token in tokens:
                           result += token + ' '
                        result += '\n'
                     result += '\n'
 
               else:
                  if (root[child][4][1].tag == 'HEADLINE'):
                     headline = root[child][4][1].text 
                     result += 'headline: ' + headline + '\n\n'
                  
                  if (root[child][4][1].tag == 'TEXT'):
                     TEXT = root[child][4][1]
                     for p in TEXT:
                        para = p.text.strip('\n')
                        sent_text = nltk.sent_tokenize(para) #Split a paragraph to one or multiple sentences.
                        for sentence in sent_text:
                           tokens = word_tokenize(sentence) #For each sentence, tokenize it.           
                           for token in tokens:
                              result += token + ' '
                           result += '\n'
                        result += '\n'

                  elif (root[child][4][2].tag == 'TEXT'):
                     TEXT = root[child][4][2]
                     for p in TEXT:
                        para = p.text.strip('\n')
                        sent_text = nltk.sent_tokenize(para) #Split a paragraph to one or multiple sentences.
                        for sentence in sent_text:
                           tokens = word_tokenize(sentence) #For each sentence, tokenize it.           
                           for token in tokens:
                              result += token + ' '
                           result += '\n'
                        result += '\n'
                  else:
                     print("different format 3")
                     pdb.set_trace()
            else:
               print("different format 2")
               pdb.set_trace()

 

   else: #TAC_share task
      root = tree.getroot()
      for child in range(len(root)):
         if (root[child].tag == 'BODY'):
            for i in range(len(root[child])):
               if (root[child][i].tag == 'HEADLINE'):
                  headline = root[child][i].text.strip('\n')
                  result += 'headline: ' + headline + '\n\n'
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
         directory = docSetA.getAttribute("id")  #Get the id for docSetA
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
   process("/dropbox/22-23/575x/Data/Documents/devtest/GuidedSumm10_test_topics.xml")

if __name__ == "__main__":
    main()