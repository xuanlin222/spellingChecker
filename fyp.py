import streamlit as st                                #--------------------------------streamlit to create the user interface
import html
import itertools
import string
from textblob import TextBlob                         #--------------------------------algorithm 1
from streamlit_option_menu import option_menu
from spellchecker import SpellChecker                 #--------------------------------algorithm 2
import re                                             #--------------------------------algorithm 2
from collections import Counter    
import time
from happytransformer import  HappyTextToText          #--------------------------------algorithm 3
from happytransformer import TTSettings                #--------------------------------algorithm 3
from fastpunct import FastPunct                        #--------------------------------algorithm 4
from neuspell import BertsclstmChecker, SclstmChecker  #--------------------------------algorithm 5
st.set_page_config(page_title='Spelling Checkers')     #--------------------------------to set the page config

#---------------------------------------------------------------------------------------create the sidebar menu
with st.sidebar:   
  selected = option_menu(
    menu_title="Main Menu",
    options=["TextBlob", "PySpell Checker", "Happy Transformer", "Fast Punct", "Sclstm Checker"],
    icons=["spellcheck", "spellcheck", "spellcheck", "spellcheck", "spellcheck"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal" #optional
  )

#---------------------------------------------------------------------------------------use textblob build in library to correct text
def algorithm1():  
  APPOSTOPHES = {
"ain't": "am not / are not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is",
"i'd": "I had / I would",
"i'd've": "I would have",
"i'll": "I shall / I will",
"i'll've": "I shall have / I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}
  slang_lookup_dict = {
    "luv" : "love", 
    "awsm" : "awesome", 
    "yrs" : "years", 
    "njhl" : "you are the best", 
    "dun" : "don't",
    "omg" : "oh my god",
    "yolo" : "you only live once",
    "bf" : "boyfriend",
    "gf" : "girlfriend",
    "bff" : "best friend forever",
    "jan" : "january",
    "feb" : "february",
    "mar" : "march",
    "apr" : "april",
    "asap" : "as soon as possible",
    "afk" : "away from keyboard",
    }

  targetSentence = st.text_area("Enter Target Sentence:", value='', height=None, max_chars=None, key=None)
  text = st.text_area("Enter Input Sentence:", value='', height=None, max_chars=None, key=None)
  if st.button('Correct Sentence'):
      if text == '':
          st.write('Please enter text for checking') 
      else: 
          startAlgorithm1 = time.time()
          decodedHtml = html.unescape(text)
          words = decodedHtml.split()
          appostophesResult = [APPOSTOPHES[word] if word in APPOSTOPHES else word for word in words]
          appostophesResult = " ".join(appostophesResult)
          punctuationsResult = appostophesResult.translate(str.maketrans('', '', string.punctuation))
          cleaned = " ".join(re.findall('[A-Z][^A-Z]*', punctuationsResult))
          slangResult = []
          for wrd in cleaned.split():
            slangResult.append(slang_lookup_dict.get(wrd, wrd))
          slangResult = ' '.join(slangResult)
          standardizedResult = ''.join(''.join(s)[:2] for _, s in itertools.groupby(slangResult))
          urlResult = TextBlob(standardizedResult)
          spellingResult = urlResult.correct()
          endAlgorithm1 = time.time()
          with st.expander("Output Sentence:"):
            st.markdown(spellingResult)
          totalTime = endAlgorithm1 - startAlgorithm1
          editDistance = damerau_levenshtein_distance(targetSentence, spellingResult)
          calculateAccuraccy(text, spellingResult, targetSentence, totalTime, editDistance)
  else: pass

#---------------------------------------------------------------------------------------use pyspellchecker build in library to correct text
def algorithm2():  
  spell = SpellChecker()
  def correct_spellings(text):
      corrected_text = []
      misspelled_words = spell.unknown(text.split())
      for word in text.split():
          if word in misspelled_words:
              corrected_text.append(spell.correction(word))
          else:
              corrected_text.append(word)
      return " ".join(filter(lambda x: str(x) if x is not None else '', corrected_text))

  targetSentence = st.text_area("Enter Target Sentence:", value='', height=None, max_chars=None, key=None)
  text = st.text_area("Enter Input Sentence:", value='', height=None, max_chars=None, key=None)
  if st.button('Correct Sentence'):
      if text == '':
          st.write('Please enter text for checking') 
      else: 
        #text = "speling correctin"
        startAlgorithm2 = time.time()
        result2 = correct_spellings(text)
        endAlgorithm2 = time.time()
        with st.expander("Output Sentence:"):
            st.markdown(result2)
        totalTime = endAlgorithm2 - startAlgorithm2
        editDistance = damerau_levenshtein_distance(targetSentence, result2)
        calculateAccuraccy(text, result2, targetSentence, totalTime, editDistance)
  else: pass

#---------------------------------------------------------------------------------------use norvig spell checker to check single word only
def algorithm():

  def words(text): return re.findall(r'\w+', text.lower())

  WORDS = Counter(words(open('C:/Users/USER/Downloads/archive/big.txt').read()))

  def P(word, N=sum(WORDS.values())): 
      "Probability of `word`."
      return WORDS[word] / N

  def correction(word): 
      "Most probable spelling correction for word."
      return max(candidates(word), key=P)

  def candidates(word): 
      "Generate possible spelling corrections for word."
      return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

  def known(words): 
      "The subset of `words` that appear in the dictionary of WORDS."
      return set(w for w in words if w in WORDS)

  def edits1(word):
      "All edits that are one edit away from `word`."
      letters    = 'abcdefghijklmnopqrstuvwxyz'
      splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
      deletes    = [L + R[1:]               for L, R in splits if R]
      transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
      replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
      inserts    = [L + c + R               for L, R in splits for c in letters]
      return set(deletes + transposes + replaces + inserts)

  def edits2(word): 
      "All edits that are two edits away from `word`."
      return (e2 for e1 in edits1(word) for e2 in edits1(e1))
  #correction('speling')
  #correction('korrectud')
  st.title('Grammar & Spell Checker In Python')
  st.write('Norvig Spell Checker')
  text = st.text_area("Enter Text:", value='', height=None, max_chars=None, key=None)
  #parser = GingerIt()
  if st.button('Correct Sentence'):
      if text == '':
          st.write('Please enter text for checking') 
      else: 
        startAlgorithm3 = time.time()
        #text = "speling correctin"
        result1 = correction(text)
        endAlgorithm3 = time.time()
        st.markdown(result1)
        st.markdown(endAlgorithm3 - startAlgorithm3)
  else: pass

#---------------------------------------------------------------------------------------use transformer build in library to correct text
def algorithm3():  
  def happytransformercheck(input):

    happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
    beam_settings =  TTSettings(num_beams=5, min_length=1, max_length=100)
    suggestions = happy_tt.generate_text(input, args=beam_settings)
    return suggestions
  targetSentence = st.text_area("Enter Target Sentence:", value='', height=None, max_chars=None, key=None)
  text = st.text_area("Enter Input Sentence:", value='', height=None, max_chars=None, key=None)
  if st.button('Correct Sentence'):
      if text == '':
          st.write('Please enter text for checking') 
      else: 
        startAlgorithm4 = time.time()
        result4 = happytransformercheck(text)
        endAlgorithm4 = time.time()
        totalTime = endAlgorithm4 - startAlgorithm4
        with st.expander("Output Sentence:"):
          st.markdown(result4.text)
        result4_1 = str(result4.text)
        editDistance = damerau_levenshtein_distance(targetSentence, result4_1)
        calculateAccuraccy(text, result4_1, targetSentence, totalTime, editDistance)

  else: pass

#---------------------------------------------------------------------------------------use fast punctuation build in library to correct text
def algorithm4():  
  def fastPunct(input):
    fastpunct = FastPunct()
    suggestions = fastpunct.punct(input)
    return suggestions
  targetSentence = st.text_area("Enter Target Sentence:", value='', height=None, max_chars=None, key=None)
  text = st.text_area("Enter Input Sentence:", value='', height=None, max_chars=None, key=None)
  if st.button('Correct Sentence'):
      if text == '':
          st.write('Please enter text for checking') 
      else: 
        startAlgorithm5 = time.time()
        result5 = fastPunct(text)
        endAlgorithm5 = time.time()
        totalTime = endAlgorithm5 - startAlgorithm5
        with st.expander("Output Sentence:"):
          st.markdown(result5)
        editDistance = damerau_levenshtein_distance(targetSentence, result5)
        calculateAccuraccy(text, result5, targetSentence, totalTime, editDistance)
  else: pass

#---------------------------------------------------------------------------------------use deep learning model to correct text
def algorithm5():  
  def sclstmChecker(input):

    checker = SclstmChecker()
    checker = checker.add_("elmo", at="input")
    checker.from_pretrained("./data/checkpoint/elmoscrnn-probwordnoise")
    output = checker.correct(input)
    return output
  targetSentence = st.text_area("Enter Target Sentence:", value='', height=None, max_chars=None, key=None)
  text = st.text_area("Enter Input Text:", value='', height=None, max_chars=None, key=None)
  if st.button('Correct Sentence'):
      if text == '':
          st.write('Please enter text for checking') 
      else: 
        startAlgorithm6 = time.time()
        result6 = sclstmChecker(text)
        endAlgorithm6 = time.time()
        totalTime = endAlgorithm6 - startAlgorithm6
       
        with st.expander("Output Sentence:"):
          st.markdown(result6)
        editDistance = damerau_levenshtein_distance(targetSentence, result6)
        calculateAccuraccy(text, result6, targetSentence, totalTime, editDistance)

  else: pass

#---------------------------------------------------------------------------------------use damerau levenshtein distance to calculate the edit distance
def damerau_levenshtein_distance(s1, s2):  
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1, lenstr1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, lenstr2 + 1):
        d[(-1, j)] = j + 1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,  # deletion
                d[(i, j - 1)] + 1,  # insertion
                d[(i - 1, j - 1)] + cost,  # substitution
            )
            if i and j and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + cost)  # transposition

    return d[lenstr1 - 1, lenstr2 - 1]

#---------------------------------------------------------------------------------------to analyse the input, output and targer sentence
def calculateAccuraccy(text, result, targetSentence, totalTime, editDistance):  
  x = 0
  y = 0
  t = 0
  try1 = 0
  try2 = 0
  test1 = 0
  lala1 = 0
  lala2 = 0
  again1 = []
  again2 = []
  again3 = []
  again4 = []
  again5 = []
  again6 = []
  lalala1 = []
  lalala2 = []
  correct = []
  wrong = [] 
#--------------------------------------------------------------------------------------remove punctuation of input, output and target sentence
  cleanedInput = text.translate(str.maketrans('', '', string.punctuation))
  punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
 
  for ele in result:
    if ele in punc:
      result = result.replace(ele, "")
  cleanedTarget = targetSentence.translate(str.maketrans('', '', string.punctuation))
  wordsInput = cleanedInput.split(' ')
  wordsResult = result.split(' ')
  wordsTarget = cleanedTarget.split(' ')

  while try1 < len(wordsResult):
    if wordsInput[try1] == wordsTarget[try1]:
      try2 += 1
      again1.append(wordsInput[try1])
      again3.append(wordsTarget[try1])
      again5.append(wordsResult[try1])
    else:
      try2 = try2
      again2.append(wordsInput[try1])
      again4.append(wordsTarget[try1])
      again6.append(wordsResult[try1])
    try1 = try1 + 1
        
  remain = len(wordsResult) - try2

  while test1 < try2:
    if again5[test1] == again3[test1]:
      lala1 += 1
    else:
      lala1 = lala1
    test1 = test1 + 1

  test1 = 0

  while test1 < remain:
    if again6[test1] == again4[test1]:
      lala2 += 1
      lalala1.append(again6[test1])
    else:
      lala2 = lala2
      lalala2.append(again6[test1])
    test1 = test1 + 1

  while x < len(wordsResult):
    if wordsResult[x] == wordsTarget[x]:
      y += 1
      correct.append(wordsTarget[x])
    else:
      y = y
      wrong.append(wordsResult[x])
    x = x + 1
#--------------------------------------------------------------------------------------to calculate the accuracy
  correct = [ele for ele in correct if ele not in lalala1]
  wrong = [ele for ele in wrong if ele not in lalala2]
  accuracy = y / len(wordsTarget) * 100
  try: 
    accuracy_t_f = 100 - (lala1/try2 * 100) #accuracy of true to false check input&target, if same=true store into try2, lala1: result&target, if same store into it
  except ZeroDivisionError as e:
     accuracy_t_f = 0
  accuracy_f_t = lala2/remain * 100 #accuracy of false to true #if correct: try2. 
  #input&target(compare): 1,3,5,7,9=same(5) ->try2[store value 5] 12
  #result&target(comapre): 1,3,5,7,9^ (check if input true to output false)=same(5) ->lala1[store value 5] 11
  #true to false --> 100 - (5/5 * 100)
  #remain=5, 
  #result&target(compare): 2,4,6,8,10=same(4) ->lala2[store value 4]
  #false to true: 4个改成对了
#--------------------------------------------------------------------------------------display the accuracy, time and edit distance
  correctString = ','.join(correct)
  wrongString = ','.join(wrong)
  lalala1String = ','.join(lalala1)
  lalala2String = ','.join(lalala2)
  with st.expander("Analysis:"):
    col1, col2, col3 = st.columns(3)
    with col1:
      st.metric('Word Correction Rate',"{:0.2f} %".format(accuracy))
      st.metric('Time per Sentence', "{:0.3f} secs".format(totalTime))
    with col2:
      st.metric('Correct To Incorrect',"{:0.2f} %".format(accuracy_t_f)) 
      st.metric('Damerau Levenshtein Distance', "{:0.0f} ".format(editDistance))
    with col3:
      st.metric('Incorrect To Correct',"{:0.2f} %".format(accuracy_f_t)) 

#--------------------------------------------------------------------------------------display the analysis result 
  with st.expander("Additional Information:"): 
    st.markdown("Correct Remain Words: ["+ correctString + "]") 
    st.markdown("Correct to Incorrect Words: ["+ wrongString + "]") 
    st.markdown("Incorrect to Correct Words: ["+ lalala1String + "]") 
    st.markdown("Incorrect to Incorrect Words: ["+ lalala2String + "]") 

#---------------------------------------------------------------------------------------user's choice
if selected == "TextBlob":
  st.title('Grammar & Spelling Checker In Python')
  st.title('Textblob using Python Library')
  algorithm1()
if selected == "PySpell Checker":
  st.title('Grammar & Spelling Checker In Python')
  st.title('Pyspellchecker Library')
  algorithm2()
if selected == "Happy Transformer":
  st.title('Grammar & Spelling Checker In Python')
  st.title('Happy Transformer')
  algorithm3()
if selected == "Fast Punct":
  st.title('Grammar & Spelling Checker In Python')
  st.title('Fast Punctuation')
  algorithm4()
if selected == "Sclstm Checker":
  st.title('Grammar & Spelling Checker In Python')
  st.title('Deep Learning Model - Sclstm Checker')
  algorithm5()

