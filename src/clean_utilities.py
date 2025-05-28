
import re
import string
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import contractions
from bs4 import BeautifulSoup
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer


# Download only what's necessary
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab')

class TextCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = SnowballStemmer("english")
        
        # Slang words
        self.sample_slang = {
            "w/e": "whatever",
            "usagov": "usa government",
            "recentlu": "recently",
            "ph0tos": "photos",
            "amirite": "am i right",
            "exp0sed": "exposed",
            "<3": "love",
            "luv": "love",
            "amageddon": "armageddon",
            "trfc": "traffic",
            "16yr": "16 year"
        }

        # Acronyms 
        self.sample_acronyms =  { 
            "mh370": "malaysia airlines flight 370",
            "okwx": "oklahoma city weather",
            "arwx": "arkansas weather",    
            "gawx": "georgia weather",  
            "scwx": "south carolina weather",  
            "cawx": "california weather",
            "tnwx": "tennessee weather",
            "azwx": "arizona weather",  
            "alwx": "alabama weather",
            "usnwsgov": "united states national weather service",
            "2mw": "tomorrow"
        }

        # Abbreviations 
        self.sample_abbr = {
            "$" : " dollar ",
            "€" : " euro ",
            "4ao" : "for adults only",
            "a.m" : "before midday",
            "a3" : "anytime anywhere anyplace",
            "aamof" : "as a matter of fact",
            "acct" : "account",
            "adih" : "another day in hell",
            "afaic" : "as far as i am concerned",
            "afaict" : "as far as i can tell",
            "afaik" : "as far as i know",
            "afair" : "as far as i remember",
            "afk" : "away from keyboard",
            "app" : "application",
            "approx" : "approximately",
            "apps" : "applications",
            "asap" : "as soon as possible",
            "asl" : "age, sex, location",
            "atk" : "at the keyboard",
            "ave." : "avenue",
            "aymm" : "are you my mother",
            "ayor" : "at your own risk", 
            "b&b" : "bed and breakfast",
            "b+b" : "bed and breakfast",
            "b.c" : "before christ",
            "b2b" : "business to business",
            "b2c" : "business to customer",
            "b4" : "before",
            "b4n" : "bye for now",
            "b@u" : "back at you",
            "bae" : "before anyone else",
            "bak" : "back at keyboard",
            "bbbg" : "bye bye be good",
            "bbc" : "british broadcasting corporation",
            "bbias" : "be back in a second",
            "bbl" : "be back later",
            "bbs" : "be back soon",
            "be4" : "before",
            "bfn" : "bye for now",
            "blvd" : "boulevard",
            "bout" : "about",
            "brb" : "be right back",
            "bros" : "brothers",
            "brt" : "be right there",
            "bsaaw" : "big smile and a wink",
            "btw" : "by the way",
            "bwl" : "bursting with laughter",
            "c/o" : "care of",
            "cet" : "central european time",
            "cf" : "compare",
            "cia" : "central intelligence agency",
            "csl" : "can not stop laughing",
            "cu" : "see you",
            "cul8r" : "see you later",
            "cv" : "curriculum vitae",
            "cwot" : "complete waste of time",
            "cya" : "see you",
            "cyt" : "see you tomorrow",
            "dae" : "does anyone else",
            "dbmib" : "do not bother me i am busy",
            "diy" : "do it yourself",
            "dm" : "direct message",
            "dwh" : "during work hours",
            "e123" : "easy as one two three",
            "eet" : "eastern european time",
            "eg" : "example",
            "embm" : "early morning business meeting",
            "encl" : "enclosed",
            "encl." : "enclosed",
            "etc" : "and so on",
            "faq" : "frequently asked questions",
            "fawc" : "for anyone who cares",
            "fb" : "facebook",
            "fc" : "fingers crossed",
            "fig" : "figure",
            "fimh" : "forever in my heart", 
            "ft." : "feet",
            "ft" : "featuring",
            "ftl" : "for the loss",
            "ftw" : "for the win",
            "fwiw" : "for what it is worth",
            "fyi" : "for your information",
            "g9" : "genius",
            "gahoy" : "get a hold of yourself",
            "gal" : "get a life",
            "gcse" : "general certificate of secondary education",
            "gfn" : "gone for now",
            "gg" : "good game",
            "gl" : "good luck",
            "glhf" : "good luck have fun",
            "gmt" : "greenwich mean time",
            "gmta" : "great minds think alike",
            "gn" : "good night",
            "g.o.a.t" : "greatest of all time",
            "goat" : "greatest of all time",
            "goi" : "get over it",
            "gps" : "global positioning system",
            "gr8" : "great",
            "gratz" : "congratulations",
            "gyal" : "girl",
            "h&c" : "hot and cold",
            "hp" : "horsepower",
            "hr" : "hour",
            "hrh" : "his royal highness",
            "ht" : "height",
            "ibrb" : "i will be right back",
            "ic" : "i see",
            "icq" : "i seek you",
            "icymi" : "in case you missed it",
            "idc" : "i do not care",
            "idgadf" : "i do not give a damn fuck",
            "idgaf" : "i do not give a fuck",
            "idk" : "i do not know",
            "ie" : "that is",
            "i.e" : "that is",
            "ifyp" : "i feel your pain",
            "IG" : "instagram",
            "iirc" : "if i remember correctly",
            "ilu" : "i love you",
            "ily" : "i love you",
            "imho" : "in my humble opinion",
            "imo" : "in my opinion",
            "imu" : "i miss you",
            "iow" : "in other words",
            "irl" : "in real life",
            "j4f" : "just for fun",
            "jic" : "just in case",
            "jk" : "just kidding",
            "jsyk" : "just so you know",
            "l8r" : "later",
            "lb" : "pound",
            "lbs" : "pounds",
            "ldr" : "long distance relationship",
            "lmao" : "laugh my ass off",
            "lmfao" : "laugh my fucking ass off",
            "lol" : "laughing out loud",
            "ltd" : "limited",
            "ltns" : "long time no see",
            "m8" : "mate",
            "mf" : "motherfucker",
            "mfs" : "motherfuckers",
            "mfw" : "my face when",
            "mofo" : "motherfucker",
            "mph" : "miles per hour",
            "mr" : "mister",
            "mrw" : "my reaction when",
            "ms" : "miss",
            "mte" : "my thoughts exactly",
            "nagi" : "not a good idea",
            "nbc" : "national broadcasting company",
            "nbd" : "not big deal",
            "nfs" : "not for sale",
            "ngl" : "not going to lie",
            "nhs" : "national health service",
            "nrn" : "no reply necessary",
            "nsfl" : "not safe for life",
            "nsfw" : "not safe for work",
            "nth" : "nice to have",
            "nvr" : "never",
            "nyc" : "new york city",
            "oc" : "original content",
            "og" : "original",
            "ohp" : "overhead projector",
            "oic" : "oh i see",
            "omdb" : "over my dead body",
            "omg" : "oh my god",
            "omw" : "on my way",
            "p.a" : "per annum",
            "p.m" : "after midday",
            "pm" : "prime minister",
            "poc" : "people of color",
            "pov" : "point of view",
            "pp" : "pages",
            "ppl" : "people",
            "prw" : "parents are watching",
            "ps" : "postscript",
            "pt" : "point",
            "ptb" : "please text back",
            "pto" : "please turn over",
            "qpsa" : "what happens", #"que pasa",
            "ratchet" : "rude",
            "rbtl" : "read between the lines",
            "rlrt" : "real life retweet", 
            "rofl" : "rolling on the floor laughing",
            "roflol" : "rolling on the floor laughing out loud",
            "rotflmao" : "rolling on the floor laughing my ass off",
            "rt" : "retweet",
            "ruok" : "are you ok",
            "sfw" : "safe for work",
            "sk8" : "skate",
            "smh" : "shake my head",
            "sq" : "square",
            "srsly" : "seriously", 
            "ssdd" : "same stuff different day",
            "tbh" : "to be honest",
            "tbs" : "tablespooful",
            "tbsp" : "tablespooful",
            "tfw" : "that feeling when",
            "thks" : "thank you",
            "tho" : "though",
            "thx" : "thank you",
            "tia" : "thanks in advance",
            "til" : "today i learned",
            "tl;dr" : "too long i did not read",
            "tldr" : "too long i did not read",
            "tmb" : "tweet me back",
            "tntl" : "trying not to laugh",
            "ttyl" : "talk to you later",
            "u" : "you",
            "u2" : "you too",
            "u4e" : "yours for ever",
            "utc" : "coordinated universal time",
            "w/" : "with",
            "w/o" : "without",
            "w8" : "wait",
            "wassup" : "what is up",
            "wb" : "welcome back",
            "wtf" : "what the fuck",
            "wtg" : "way to go",
            "wtpa" : "where the party at",
            "wuf" : "where are you from",
            "wuzup" : "what is up",
            "wywh" : "wish you were here",
            "yd" : "yard",
            "ygtr" : "you got that right",
            "ynk" : "you never know",
            "zzz" : "sleeping bored and tired"
        }        
        
    def clean_text(self, text, remove_stopwords=True):
        """Optimized text cleaning for embeddings and LSTM"""
        if pd.isna(text) or text == "":
            return ""
        
        # STEP 1: Remove HTML and XML tags
        text = self._remove_html_xml(text)
        
        # STEP 2: Remove non-ASCII characters
        text = self.remove_non_ascii(text)
        
        # STEP 3: Fix contractions 
        text = self.fix_contractions(text)
        
        # STEP 4: Replace entities with placeholders
        text = self.replace_special_entities(text)
        
        # STEP 5: Convert to lowercase
        text = text.lower()
        
        # STEP 6: Typos, slang, Acronyms, Some common abbreviations
        text = self.other_clean(text)
        
        # STEP 7: Remove punctuation - embeddings typically ignore punctuation
        text = self.remove_punctuation(text)
        
        # STEP 8: Remove numbers
        text = self.remove_numbers(text)
        
        # STEP 9: Remove extra whitespace
        text = ' '.join(text.split())
        
        # STEP 10: Tokenize
        tokens = word_tokenize(text)
        
        # STEP 11: Remove stop words (optional - depends on your specific task)
        if remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]
        
        # STEP 12: Stemming
        # tokens = self._lancaster_stemmer(tokens)
        # tokens = self._snowball_stemmer(tokens)
        
        # Return joined tokens
        return ' '.join(tokens)
    
    def _remove_html_xml(self, text):
        """Remove HTML and XML tags"""
        return BeautifulSoup(text, "html.parser").get_text()
    
    def remove_non_ascii(self, text):
        """Remove non-ASCII characters"""
        return re.sub(r'[^\x00-\x7f]',r'', text) # or ''.join([x for x in text if x in string.printable])
    
    def fix_contractions(self, text):
        """Expand contractions (e.g., don't -> do not)"""
        return contractions.fix(text)
    
    def replace_special_entities(self, text):
        """Simplified entity replacement - focusing on most common patterns"""
        
        # URLs - important for social media data
        text = re.sub(r'https?://\S+|www\.\S+', 'URL', text)
        
        # Emails
        text = re.sub(r'\S+@\S+', 'EMAIL', text)
        
        # Simplified date replacement - covering major formats
        text = re.sub(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}', 'DATE', text)
        text = re.sub(r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}', 'DATE', text, flags=re.IGNORECASE)
        text = re.sub(r'\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4}', 'DATE', text, flags=re.IGNORECASE)
        
        # Simplified time replacement
        text = re.sub(r'\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]m)?', 'TIME', text, flags=re.IGNORECASE)
        
        return text
    
    def remove_punctuation(self,text):
        """Remove the punctuation"""
    #     return re.sub(r'[]!"$%&\'()*+,./:;=#@?[\\^_`{|}~-]+', "", text)
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def remove_numbers(self, text):
        """Remove digits using regular expressions"""
        return re.sub(r'\d+', '', text)
    
    def snowball_stemmer(self, text):
        """Stem words in list of tokenized words with SnowballStemmer"""
        stems = [self.stemmer.stem(i) for i in text]
        return stems

    def lancaster_stemmer(self, text):
        """
            Stem words in list of tokenized words with LancasterStemmer
        """
        stemmer = LancasterStemmer()
        stems = [stemmer.stem(i) for i in text]
        return stems
    
    def process_dataframe(self, df, text_column, batch_size=5000):
        """Process dataframe in batches for better memory management"""
        total_rows = len(df)
        cleaned_texts = []
        
        for i in range(0, total_rows, batch_size):
            batch = df[text_column].iloc[i:i+batch_size]
            batch_cleaned = batch.apply(self.clean_text)
            cleaned_texts.extend(batch_cleaned.tolist())
            # Print progress every 10 batches
            if (i // batch_size) % 2 == 0:
                print(f"Processed {i+len(batch)}/{total_rows} texts ({((i+len(batch))/total_rows)*100:.1f}%)")
        
        return cleaned_texts

    
    def other_clean(self, text):
        """
            Other manual text cleaning techniques
        """ 
        sample_typos_slang_pattern = re.compile(r'(?<!\w)(' + '|'.join(re.escape(key) for key in self.sample_slang.keys()) + r')(?!\w)')
        sample_acronyms_pattern = re.compile(r'(?<!\w)(' + '|'.join(re.escape(key) for key in self.sample_acronyms.keys()) + r')(?!\w)')
        sample_abbr_pattern = re.compile(r'(?<!\w)(' + '|'.join(re.escape(key) for key in self.sample_abbr.keys()) + r')(?!\w)')
        
        text = sample_typos_slang_pattern.sub(lambda x: self.sample_slang[x.group()], text)
        text = sample_acronyms_pattern.sub(lambda x: self.sample_acronyms[x.group()], text)
        text = sample_abbr_pattern.sub(lambda x: self.sample_abbr[x.group()], text)
        
        return text