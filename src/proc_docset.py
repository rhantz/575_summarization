from xml.dom.minidom import parse
import xml.dom.minidom
import pdb
import os
import xml.etree.ElementTree as ET
from lxml import etree
import nltk
nltk.download('punkt')
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
   corpora = '1' # 1:AQUAINT-2, 2:AQUAINT, 3:TAC 2011 shared task

   doc_id_length = len(doc_id)
   if (doc_id_length == 16):    #AQUAINT
      dir = '/corpora/LDC/LDC02T31/'
      corpora = '2'
   elif (doc_id_length == 21): #AQUAINT-2
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


   if (corpora == '1'):
      try: #AQUAINT-2
         f = open(dir + '/' + articles, 'r')            
         parser = etree.XMLParser(recover=True)
         tree = etree.parse(f, parser)
      except:
         try:
            f = open(dir + '/' + articles + '.xml','r')               
            parser = etree.XMLParser(recover=True)
            tree = etree.parse(f, parser)
         except:
            corpora = '3'
   if (corpora == '2'):
      try: #AQUAINT
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
      except:
         corpora = '3'

   if (corpora == '3'): #TAC 2011 shared task
      try:
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
      try:
         root = tree.getroot()
      except:
         print("local variable 'tree' referenced before assignment")
         pdb.set_trace()
      DOCs = root.findall('DOC')
      if (len(DOCs) > 0):
         for DOC in DOCs:
            DOC_id = DOC.attrib['id']
            if (DOC_id == doc_id):                     
               HEADLINE = DOC.findall('HEADLINE')
               if (len(HEADLINE) > 0):
                  headline = HEADLINE[0].text.strip('\n')
                  result += 'headline: ' + headline + '\n\n'
               DATELINE = DOC.findall('DATELINE')
               if (len(DATELINE) > 0):
                  dateline = DATELINE[0].text.strip('\n')
                  result += 'date-time: ' + dateline + '\n\n'
               TEXT = DOC.findall('TEXT')
               if (len(TEXT[0]) == 0): #no <p>. Content stored directly in <TEXT>                    
                  for para in TEXT[0].text.strip('\n').split('\n\t'):
                     sent_text = nltk.sent_tokenize(para) #Split a paragraph to one or multiple sentences.
                     for sentence in sent_text:
                        tokens = word_tokenize(sentence) #For each sentence, tokenize it.           
                        for token in tokens:
                           result += token + ' '
                        result += '\n'
                  result += '\n'
               else:
                  for p in TEXT[0]:
                     para = p.text.strip('\n')
                     sent_text = nltk.sent_tokenize(para) #Split a paragraph to one or multiple sentences.
                     for sentence in sent_text:
                        tokens = word_tokenize(sentence) #For each sentence, tokenize it.           
                        for token in tokens:
                           result += token + ' '
                        result += '\n'
                     result += '\n'
          
      else:
         print('no DOC')
         pdb.set_trace() 


   elif (corpora == '2'): #1996-2000  
      '''
      DOMTree = xml.dom.minidom.parse(dir + '/' + articles)
      collection = DOMTree.documentElement
      topics = collection.getElementsByTagName("topic")
      '''
      # Having trouble with this part cause the xml files have very different format, solved it by using .findall() function
      root = tree.getroot()
      DOCs = root.findall('DOC')
      if (len(DOCs) > 0):
         for DOC in DOCs:
            DOC_id = DOC.findall('DOCNO')
            if (len(DOC_id) == 1):               
               if (doc_id in DOC_id[0].text):
                  DATE_TIME = DOC.findall('DATE_TIME')
                  BODY = DOC.findall('BODY')
                  HEADLINE = BODY[0].findall('HEADLINE')
                  if (len(HEADLINE) > 0):
                     headline = HEADLINE[0].text.strip('\n')
                     result += 'headline: ' + headline + '\n\n'
                  if (len(DATE_TIME) > 0):
                     dateline = DATE_TIME[0].text.strip('\n')
                     result += 'date-time: ' + dateline + '\n\n'
                  TEXT = BODY[0].findall('TEXT')
                  if (len(TEXT[0]) == 0): #no <p>. Content stored directly in <TEXT>                    
                     for para in TEXT[0].text.strip('\n').split('\n\t'):
                        sent_text = nltk.sent_tokenize(para) #Split a paragraph to one or multiple sentences.
                        for sentence in sent_text:
                           tokens = word_tokenize(sentence) #For each sentence, tokenize it.           
                           for token in tokens:
                              result += token + ' '
                           result += '\n'
                        result += '\n'
                  for p in TEXT[0]:
                        para = p.text.strip('\n')
                        sent_text = nltk.sent_tokenize(para) #Split a paragraph to one or multiple sentences.
                        for sentence in sent_text:
                           tokens = word_tokenize(sentence) #For each sentence, tokenize it.           
                           for token in tokens:
                              result += token + ' '
                           result += '\n'
                        result += '\n'

            elif (len(DOC_id) > 1):
               print("more than one DOCNO")
               pdb.set_trace()
            else:
               print("no DOCNO in DOC")
               pdb.set_trace()
          
      else:
         print('no DOC')
         pdb.set_trace() 



   else: #TAC_share task
      root = tree.getroot()
      for child in range(len(root)):
         if (root[child].tag == 'DATETIME'):
            DATELINE = root[child].text.strip('\n')
         if (root[child].tag == 'BODY'):
            for i in range(len(root[child])):
               if (root[child][i].tag == 'HEADLINE'):
                  headline = root[child][i].text.strip('\n')
                  result += 'headline: ' + headline + '\n\n'
                  try:
                     result += 'date-time: ' + DATELINE + '\n\n'
                  except:
                     pass
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