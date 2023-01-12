from xml.dom.minidom import parse
import xml.dom.minidom
import pdb
import os
import xml.etree.ElementTree as ET


'''
Official Documents for xml.dom:
https://docs.python.org/3/library/xml.dom.html#dom-nodelist-objects

An demo of xml.dom (in Chinese):
https://www.runoob.com/python/python-xml.html
'''

def splitParagraph(doc_id):
   '''
   We will first find the article based on the doc id. Articles published in 1996-2000 is in /corpora/LDC/LDC02T31/.
   Articles published in 2004-2006 is in /corpora/LDC/LDC08T25/data/
   '''
   dir = ''
   year = int(doc_id[8:12]) #e.g. '2004'
   if (1996 <= year <= 2000):
      dir = '/corpora/LDC/LDC02T31/'
   elif (2004 <= year <= 2006):
      dir = '/corpora/LDC/LDC08T25/data/'
   else:
      print("error, not in any dataset")
      pdb.set_trace()
   
   subdir = ""
   docType = doc_id[0:7].lower() #e.g. 'xin_eng'
   dir += docType 
   articles = doc_id[0:14].lower() #e.g. 'xin_eng_200411'

   # DOMTree = xml.dom.minidom.parse(dir + '/' + articles + '.xml')
   # collection = DOMTree.documentElement
   # docs = collection.getElementsByTagName("DOC")

   tree = ET.parse(dir + '/' + articles + '.xml')
   article = tree.findall('DOC[@id="'+doc_id+'"]')
   #Problem: how to read headline and paragraph after finding the doc_id?
   pdb.set_trace()
   return

def tokenization():
   return


# read the .xml file using minidom parser
DOMTree = xml.dom.minidom.parse("/dropbox/22-23/575x/Data/Documents/training/2009/UpdateSumm09_test_topics.xml")
collection = DOMTree.documentElement

topics = collection.getElementsByTagName("topic")

for topic in topics:
   print ("*****Topic*****")
   topic_id = topic.getAttribute("id")
   title = topic.getElementsByTagName("title")[0]
   print ("title: %s" % title.childNodes[0].data)
   narrative = topic.getElementsByTagName("narrative")[0]
   print ("narrative: %s" % narrative.childNodes[0].data)
   docsetAs = topic.getElementsByTagName("docsetA")

   for docSetA in docsetAs:
      # For each docSetA, we will create a directory under output_dir. The name of the subdirectory is the same as the docsetA id.
      directory = docSetA.getAttribute("id")
      path = '../outputs/training/'+directory
      if not os.path.exists(path):
         os.makedirs(path)
      doc = docSetA.getElementsByTagName("doc")
      length = doc.length
      for i in range(length):
         print ("doc: %s" % doc.item(i).getAttribute("id"))
         doc_id = doc.item(i).getAttribute("id")
         # Open '../outputs/training/D0901A-AXIN_ENG_20041113.0001'
         output_file = open(path + '/' + doc_id, "w") 
         splitParagraph(doc_id)
         tokenization()
         output_file.writelines(doc_id)
         output_file.close()