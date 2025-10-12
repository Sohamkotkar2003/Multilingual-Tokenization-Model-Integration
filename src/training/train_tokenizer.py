"""
SentencePiece Tokenizer Training Script for Multilingual Support

This script trains a custom SentencePiece tokenizer on 21 Indian languages:
Assamese, Bengali, Bodo, English, Gujurati, Hindi, Kannada, Kashmiri, Maithili,
Malyalam, Marathi, Meitei, Nepali, Odia, Punjabi, Sanskrit, Santali, Sindhi,
Tamil, Telugu, and Urdu with proper handling of various Indic scripts and ligatures.

Usage:
    python train_tokenizer.py

The script will:
1. Collect training data from all language files
2. Apply proper Unicode normalization for all Indic scripts
3. Train a BPE tokenizer with appropriate vocabulary size
4. Save the tokenizer model for use with the API

Requirements:
    - Training data files in data/training/ directory
    - sentencepiece library installed
"""

import os
import logging
import unicodedata
import sentencepiece as spm
from typing import List, Dict
import tempfile
from config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultilingualTokenizerTrainer:
    def __init__(self):
        self.temp_combined_file = None
        self.stats = {
            "total_sentences": 0,
            "language_counts": {},
            "devanagari_sentences": 0,
            "latin_sentences": 0
        }

    def normalize_devanagari_text(self, text: str) -> str:
        """
        Apply proper Unicode normalization for Devanagari text
        This ensures consistent handling of ligatures and combining characters
        Note: We do NFC normalization in Python since SentencePiece NFC normalization
        has compatibility issues on Windows
        """
        # Apply NFC normalization to combine base characters with diacritics
        normalized = unicodedata.normalize('NFC', text)
        
        # Remove zero-width characters that can cause tokenization issues
        zero_width_chars = ['\u200b', '\u200c', '\u200d', '\ufeff']
        for char in zero_width_chars:
            normalized = normalized.replace(char, '')
        
        return normalized.strip()

    def detect_script(self, text: str) -> str:
        """Detect the primary script of the text"""
        devanagari_count = sum(1 for c in text if 
                             settings.DEVANAGARI_UNICODE_RANGE[0] <= ord(c) <= settings.DEVANAGARI_UNICODE_RANGE[1])
        latin_count = sum(1 for c in text if c.isascii() and c.isalpha())
        total_alpha = sum(1 for c in text if c.isalpha())
        
        if total_alpha == 0:
            return "mixed"
        
        if devanagari_count / total_alpha > 0.5:
            return "devanagari"
        elif latin_count / total_alpha > 0.5:
            return "latin"
        else:
            return "mixed"

    def prepare_training_data(self) -> str:
        """
        Collect and prepare training data from all language files
        Returns: Path to the combined training file
        """
        logger.info("Preparing multilingual training data...")
        
        # Create temporary file for combined training data
        temp_fd, self.temp_combined_file = tempfile.mkstemp(suffix='.txt', prefix='multilingual_training_')
        
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as outfile:
            for lang, filename in settings.CORPUS_FILES.items():
                filepath = os.path.join(settings.TRAINING_DATA_PATH, filename)
                
                if not os.path.exists(filepath):
                    logger.warning(f"Training file not found: {filepath}")
                    
                    # Create sample data if file doesn't exist
                    sample_data = self.create_sample_data(lang)
                    os.makedirs(settings.TRAINING_DATA_PATH, exist_ok=True)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(sample_data)
                    logger.info(f"Created sample data for {lang}")
                
                # Process the file
                lang_sentence_count = 0
                with open(filepath, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        line = line.strip()
                        if len(line) < 10:  # Skip very short lines
                            continue
                        
                        # Apply normalization (especially important for Devanagari)
                        normalized_line = self.normalize_devanagari_text(line)
                        
                        if len(normalized_line) < 10:
                            continue
                        
                        # Write to combined file
                        outfile.write(normalized_line + '\n')
                        lang_sentence_count += 1
                        self.stats["total_sentences"] += 1
                        
                        # Update script statistics
                        script = self.detect_script(normalized_line)
                        if script == "devanagari":
                            self.stats["devanagari_sentences"] += 1
                        elif script == "latin":
                            self.stats["latin_sentences"] += 1
                
                self.stats["language_counts"][lang] = lang_sentence_count
                logger.info(f"Loaded {lang_sentence_count} sentences from {lang}")
        
        logger.info(f"Total training sentences: {self.stats['total_sentences']}")
        logger.info(f"Devanagari sentences: {self.stats['devanagari_sentences']}")
        logger.info(f"Latin sentences: {self.stats['latin_sentences']}")
        
        return self.temp_combined_file

    def create_sample_data(self, language: str) -> str:
        """Create sample training data if files don't exist"""
        sample_data = {
            "assamese": """
নমস্কাৰ, আপুনি কেনে আছে?
মই অসমীয়া ভাষা শিকি আছোঁ।
ভাৰত এখন সুন্দৰ দেশ।
শিক্ষা অতি গুৰুত্বপূৰ্ণ।
আমি আমাৰ ভাষা আৰু সংস্কৃতি সংৰক্ষণ কৰিব লাগে।
প্ৰকৃতিৰ সৌন্দৰ্য অপূৰ্ব।
বিজ্ঞান আৰু প্ৰযুক্তি বিকাশৰ চালিকা শক্তি।
কঠোৰ পৰিশ্ৰমে সফলতা আনে।
পৰিৱেশ সংৰক্ষণ অতি জৰুৰী।
সকলোৱে শিক্ষাৰ অধিকাৰ পাব লাগে।
            """.strip(),
            
            "bengali": """
নমস্কার, আপনি কেমন আছেন?
আমি বাংলা ভাষা শিখছি।
ভারত একটি সুন্দর দেশ।
শিক্ষা সবচেয়ে গুরুত্বপূর্ণ।
আমাদের ভাষা এবং সংস্কৃতি রক্ষা করা উচিত।
প্রকৃতির সৌন্দর্য অপূর্ব।
বিজ্ঞান এবং প্রযুক্তি উন্নয়নের চালিকা শক্তি।
কঠোর পরিশ্রম সাফল্য আনে।
পরিবেশ সংরক্ষণ অত্যন্ত জরুরি।
সবার শিক্ষার অধিকার থাকা উচিত।
            """.strip(),
            
            "bodo": """
नमस्कार, नों मिथिंनाय दं?
आं बड़ो भाषा सिखनाय।
भारत गोबां सुन्दर देश।
सिक्षा बेयो महत्वाको थानाय।
लांनि भाषा आरो संस्कृति रक्सा खालामनाय जायो।
फुरान्नि सुन्दर गोसाइ अफादगोन।
बिजान आरो टेक्नोलोजी दावखि खालाम।
गोसै सिराय खुरा फायदा होनाय।
फुरानखों रक्सा खालाम गोनां महत्वाको।
गासै सिक्साफोरनि अधिकार दङो।
            """.strip(),
            
            "english": """
Hello, how are you today?
Learning multiple languages is beneficial.
Education is the foundation of progress.
Technology has transformed our lives.
Reading books expands our knowledge.
Environmental protection is crucial.
Cultural diversity makes us stronger.
Hard work leads to success.
Respect for others is important.
Science and innovation drive development.
            """.strip(),
            
            "gujurati": """
નમસ્તે, તમે કેવા છો?
હું ગુજરાતી ભાષા શીખી રહ્યો છું।
ભારત એક સુંદર દેશ છે।
શિક્ષણ સૌથી મહત્વપૂર્ણ છે।
આપણે આપણી ભાષા અને સંસ્કૃતિનું રક્ષણ કરવું જોઈએ।
પ્રકૃતિની સુંદરતા અદ્ભુત છે।
વિજ્ઞાન અને ટેકનોલોજી વિકાસને ચલાવે છે।
મહેનત સફળતા લાવે છે।
પર્યાવરણ સંરક્ષણ અત્યંત જરૂરી છે।
દરેકને શિક્ષણનો અધિકાર હોવો જોઈએ।
            """.strip(),
            
            "hindi": """
नमस्ते, आप कैसे हैं?
मैं हिंदी भाषा सीख रहा हूं।
भारत एक बहुत सुंदर देश है।
यह पुस्तक बहुत अच्छी है।
हमें अपनी भाषा पर गर्व होना चाहिए।
शिक्षा सबसे महत्वपूर्ण है।
स्वतंत्रता दिवस एक राष्ट्रीय त्योहार है।
गणित एक कठिन विषय हो सकता है।
प्रकृति की सुंदरता अद्भुत है।
हमें पर्यावरण की रक्षा करनी चाहिए।
            """.strip(),
            
            "kannada": """
ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?
ನಾನು ಕನ್ನಡ ಭಾಷೆಯನ್ನು ಕಲಿಯುತ್ತಿದ್ದೇನೆ।
ಭಾರತ ಒಂದು ಸುಂದರ ದೇಶ।
ಶಿಕ್ಷಣವು ಅತ್ಯಂತ ಮಹತ್ವದ್ದು।
ನಾವು ನಮ್ಮ ಭಾಷೆ ಮತ್ತು ಸಂಸ್ಕೃತಿಯನ್ನು ರಕ್ಷಿಸಬೇಕು।
ಪ್ರಕೃತಿಯ ಸೌಂದರ್ಯ ಅದ್ಭುತವಾಗಿದೆ।
ವಿಜ್ಞಾನ ಮತ್ತು ತಂತ್ರಜ್ಞಾನ ಅಭಿವೃದ್ಧಿಯನ್ನು ಚಾಲನೆ ಮಾಡುತ್ತವೆ।
ಕಷ್ಟಪಟ್ಟು ದುಡಿದರೆ ಯಶಸ್ಸು ದೊರೆಯುತ್ತದೆ।
ಪರಿಸರ ಸಂರಕ್ಷಣೆ ಅತ್ಯಂತ ಅಗತ್ಯ।
ಎಲ್ಲರಿಗೂ ಶಿಕ್ಷಣದ ಹಕ್ಕಿರಬೇಕು।
            """.strip(),
            
            "kashmiri": """
نمسکار، تۄہہ کیتھ چھیو؟
بٕہ کٲشُر زَبان ہیکان چھُس۔
بارَت اکھ خوبصورت مُلک چھُ۔
تٲلیم سبنٕ کھۄتۍ اہم چھُ۔
أسۍ پننۍ زَبان تہٕ ثقافت محفوظ رٔکھنۍ پکار چھُ۔
قُدرتُک خوبصورتی عجیب چھُ۔
سائنس تہٕ ٹیکنالوجی ترقیٔ چلاوان چھُ۔
محنت کامیابی آنان چھُ۔
ماحولیات تحفظ بوزِ ضروری چھُ۔
سبنن تٲلیمُک حق آسُن پکار چھُ।
            """.strip(),
            
            "maithili": """
नमस्कार, अहाँ केहन छी?
हम मैथिली भाषा सीखि रहल छी।
भारत एकटा सुन्दर देश छै।
शिक्षा सबसँ महत्वपूर्ण अछि।
हमरा सभकेँ अपन भाषा आ संस्कृतिक रक्षा कएनाइ चाही।
प्रकृतिक सुन्दरता अद्भुत अछि।
विज्ञान आ प्रौद्योगिकी विकास चलबैत अछि।
मेहनतिसँ सफलता भेटैत अछि।
पर्यावरण संरक्षण बहुत जरूरी अछि।
सभकेँ शिक्षाक अधिकार होबाक चाही।
            """.strip(),
            
            "malyalam": """
നമസ്കാരം, നിങ്ങൾക്ക് എങ്ങനെയുണ്ട്?
ഞാൻ മലയാളം ഭാഷ പഠിക്കുകയാണ്।
ഭാരതം മനോഹരമായ ഒരു രാജ്യമാണ്।
വിദ്യാഭ്യാസം ഏറ്റവും പ്രധാനപ്പെട്ടതാണ്।
നമ്മുടെ ഭാഷയും സംസ്കാരവും സംരക്ഷിക്കണം।
പ്രകൃതിയുടെ സൗന്ദര്യം അത്ഭുതകരമാണ്।
ശാസ്ത്രവും സാങ്കേതികവിദ്യയും വികസനത്തെ നയിക്കുന്നു।
കഠിനാധ്വാനം വിജയം നൽകുന്നു।
പരിസ്ഥിതി സംരക്ഷണം അത്യാവശ്യമാണ്।
എല്ലാവർക്കും വിദ്യാഭ്യാസത്തിനുള്ള അവകാശം ഉണ്ടായിരിക്കണം।
            """.strip(),
            
            "marathi": """
नमस्कार, तुम्ही कसे आहात?
मराठी ही महाराष्ट्राची भाषा आहे।
मला मराठी भाषा आवडते।
शिक्षण हा सर्वात महत्वाचा मुद्दा आहे।
पुस्तक वाचणे हा चांगला सवय आहे।
निसर्गाचे संरक्षण करणे आवश्यक आहे।
सणउत्सव आपल्या संस्कृतीचा भाग आहेत।
कष्ट केल्याशिवाय काहीही मिळत नाही।
एकता आणि अखंडता महत्वपूर्ण आहे।
सत्य आणि न्याय हेच खरे मूल्य आहेत।
            """.strip(),
            
            "meitei": """
নমস্কার, নহাক কৈদোক্লেবগা?
ঐখোয় মৈতৈলোন্ ততুরি।
ভারত ইকাই ফজবা লৈবাক নি।
এদুকায়ন্না খ্বাইদগি মরুওইবা নি।
ঐখোয়না ঐখোয়গি লোন্ অমসুঙ কল্চর্ দু য়েনশিনগদবনি।
নেচর্গি ফজরবা শকপি চাওখত্তবা ওইখি।
সায়েন্স অমসুঙ টেকনোলোজিনা দেভেলপমেন্টবু য়াইহন্লি।
ৱাখল থবকনা ফজনবা পুরকহনবা ওইহনি।
এন্ভাইরোনমেন্ট প্রোটেক্সন খ্বাইদগি মরুওইবা নি।
পুম্নমক্না এদুকেসনগি মতিক লৈরি।
            """.strip(),
            
            "nepali": """
नमस्कार, तपाईं कस्तो हुनुहुन्छ?
म नेपाली भाषा सिक्दै छु।
भारत एउटा सुन्दर देश हो।
शिक्षा सबैभन्दा महत्त्वपूर्ण छ।
हामीले हाम्रो भाषा र संस्कृति जोगाउनुपर्छ।
प्रकृतिको सुन्दरता अद्भुत छ।
विज्ञान र प्रविधिले विकास चलाउँछ।
मेहनतले सफलता ल्याउँछ।
वातावरण संरक्षण अत्यन्त आवश्यक छ।
सबैलाई शिक्षाको अधिकार हुनुपर्छ।
            """.strip(),
            
            "odia": """
ନମସ୍କାର, ଆପଣ କେମିତି ଅଛନ୍ତି?
ମୁଁ ଓଡ଼ିଆ ଭାଷା ଶିଖୁଛି।
ଭାରତ ଏକ ସୁନ୍ଦର ଦେଶ।
ଶିକ୍ଷା ସବୁଠାରୁ ଗୁରୁତ୍ୱପୂର୍ଣ୍ଣ।
ଆମେ ଆମର ଭାଷା ଏବଂ ସଂସ୍କୃତିକୁ ରକ୍ଷା କରିବା ଉଚିତ।
ପ୍ରକୃତିର ସୌନ୍ଦର୍ଯ୍ୟ ଅଦ୍ଭୁତ।
ବିଜ୍ଞାନ ଏବଂ ପ୍ରଯୁକ୍ତି ବିକାଶକୁ ଚାଳନା କରେ।
କଠିନ ପରିଶ୍ରମ ସଫଳତା ଆଣେ।
ପରିବେଶ ସଂରକ୍ଷଣ ଅତ୍ୟନ୍ତ ଆବଶ୍ୟକ।
ସମସ୍ତଙ୍କର ଶିକ୍ଷାର ଅଧିକାର ରହିବା ଉଚିତ।
            """.strip(),
            
            "punjabi": """
ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?
ਮੈਂ ਪੰਜਾਬੀ ਭਾਸ਼ਾ ਸਿੱਖ ਰਿਹਾ ਹਾਂ।
ਭਾਰਤ ਇੱਕ ਸੁੰਦਰ ਦੇਸ਼ ਹੈ।
ਸਿੱਖਿਆ ਸਭ ਤੋਂ ਮਹੱਤਵਪੂਰਨ ਹੈ।
ਸਾਨੂੰ ਆਪਣੀ ਭਾਸ਼ਾ ਅਤੇ ਸੱਭਿਆਚਾਰ ਦੀ ਰੱਖਿਆ ਕਰਨੀ ਚਾਹੀਦੀ ਹੈ।
ਕੁਦਰਤ ਦੀ ਸੁੰਦਰਤਾ ਸ਼ਾਨਦਾਰ ਹੈ।
ਵਿਗਿਆਨ ਅਤੇ ਤਕਨਾਲੋਜੀ ਵਿਕਾਸ ਨੂੰ ਚਲਾਉਂਦੇ ਹਨ।
ਮਿਹਨਤ ਸਫਲਤਾ ਲਿਆਉਂਦੀ ਹੈ।
ਵਾਤਾਵਰਣ ਸੁਰੱਖਿਆ ਬਹੁਤ ਜ਼ਰੂਰੀ ਹੈ।
ਸਾਰਿਆਂ ਨੂੰ ਸਿੱਖਿਆ ਦਾ ਅਧਿਕਾਰ ਹੋਣਾ ਚਾਹੀਦਾ ਹੈ।
            """.strip(),
            
            "sanskrit": """
नमस्कार, कथं वर्तसे?
संस्कृतं भारतस्य प्राचीनतमा भाषा अस्ति।
वेदाः संस्कृते लिखिताः सन्ति।
धर्मो रक्षति रक्षितः।
सत्यं शिवं सुन्दरम्।
विद्या ददाति विनयं।
यत्र नार्यस्तु पूज्यन्ते रमन्ते तत्र देवताः।
सर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः।
वसुधैव कुटुम्बकम्।
अहिंसा परमो धर्मः।
            """.strip(),
            
            "santali": """
ᱡᱚᱦᱟᱨ, ᱟᱢ ᱪᱮᱫᱮ ᱢᱮᱱᱟᱢᱟ?
ᱤᱧ ᱥᱟᱱᱛᱟᱲᱤ ᱯᱟᱹᱨᱥᱤ ᱥᱮᱪᱮᱫ ᱠᱟᱱᱟᱧ।
ᱵᱷᱟᱨᱚᱛ ᱫᱚ ᱢᱤᱫᱴᱟᱹᱝ ᱡᱟᱹᱥᱛᱤ ᱪᱚᱨᱚᱠ ᱫᱤᱥᱚᱢ ᱠᱟᱱᱟ।
ᱥᱮᱪᱮᱫ ᱫᱚ ᱟᱹᱰᱤ ᱞᱟᱹᱠᱛᱤᱭᱟᱱᱟ।
ᱟᱞᱮ ᱟᱞᱮᱭᱟᱜ ᱯᱟᱹᱨᱥᱤ ᱟᱨ ᱞᱟᱠᱪᱟᱨ ᱫᱚᱦᱚ ᱠᱟᱛᱮ ᱫᱚᱦᱚ ᱦᱚᱨᱚ ᱦᱚᱨᱚ ᱫᱚᱦᱚ।
ᱡᱟᱹᱱᱩᱢ ᱪᱟᱹᱨᱤᱛ ᱟᱹᱰᱤ ᱪᱚᱨᱚᱠ ᱜᱮᱭᱟ।
ᱥᱟᱬᱮᱥ ᱟᱨ ᱴᱮᱠᱱᱳᱞᱳᱡᱤ ᱫᱚ ᱞᱟᱦᱟᱱᱛᱤ ᱟᱹᱪᱩᱨᱟ।
ᱪᱮᱫ ᱠᱟᱹᱢᱤ ᱥᱟᱢᱟᱝ ᱟᱹᱜᱩᱣᱟ।
ᱣᱟᱛᱟᱣᱨᱚᱬ ᱫᱚᱦᱚ ᱟᱹᱰᱤ ᱞᱟᱹᱠᱛᱤᱭᱟᱱᱟ।
ᱡᱚᱛᱚ ᱦᱚᱲ ᱠᱚ ᱥᱮᱪᱮᱫ ᱧᱟᱢ ᱦᱚᱠ ᱛᱟᱦᱮᱸᱱᱟ।
            """.strip(),
            
            "sindhi": """
سلام، توهان ڪيئن آهيو؟
مان سنڌي ٻولي سکي رهيو آهيان।
ڀارت هڪ خوبصورت ملڪ آهي।
تعليم سڀ کان اهم آهي।
اسان کي پنهنجي ٻولي ۽ ثقافت جي حفاظت ڪرڻ گهرجي।
فطرت جي خوبصورتي شاندار آهي।
سائنس ۽ ٽيڪنالاجي ترقي کي هلائي رهيا آهن।
محنت ڪاميابي ڏياري رهي آهي।
ماحولياتي تحفظ تمام ضروري آهي।
سڀني کي تعليم جو حق هجڻ گهرجي।
            """.strip(),
            
            "tamil": """
வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?
நான் தமிழ் மொழி கற்றுக்கொண்டிருக்கிறேன்।
இந்தியா ஒரு அழகான நாடு।
கல்வி மிக முக்கியமானது।
நாம் நமது மொழி மற்றும் பண்பாட்டைப் பாதுகாக்க வேண்டும்।
இயற்கையின் அழகு அற்புதமானது।
அறிவியல் மற்றும் தொழில்நுட்பம் வளர்ச்சியை இயக்குகிறது।
கடின உழைப்பு வெற்றியைத் தருகிறது।
சுற்றுச்சூழல் பாதுகாப்பு மிகவும் அவசியம்।
அனைவருக்கும் கல்வி உரிமை இருக்க வேண்டும்।
            """.strip(),
            
            "telugu": """
నమస్కారం, మీరు ఎలా ఉన్నారు?
నేను తెలుగు భాష నేర్చుకుంటున్నాను।
భారతదేశం అందమైన దేశం।
విద్య చాలా ముఖ్యం।
మనం మన భాష మరియు సంస్కృతిని కాపాడుకోవాలి।
ప్రకృతి సౌందర్యం అద్భుతమైనది।
విజ్ఞానం మరియు సాంకేతికత అభివృద్ధిని నడిపిస్తాయి।
కష్టపడి పనిచేస్తే విజయం వస్తుంది।
పర్యావరణ పరిరక్షణ చాలా అవసరం।
అందరికీ విద్య హక్కు ఉండాలి।
            """.strip(),
            
            "urdu": """
السلام علیکم، آپ کیسے ہیں؟
میں اردو زبان سیکھ رہا ہوں۔
بھارت ایک خوبصورت ملک ہے۔
تعلیم سب سے اہم ہے۔
ہمیں اپنی زبان اور ثقافت کی حفاظت کرنی چاہیے۔
قدرت کی خوبصورتی حیرت انگیز ہے۔
سائنس اور ٹیکنالوجی ترقی کو چلاتے ہیں۔
محنت سے کامیابی ملتی ہے۔
ماحولیات کی حفاظت انتہائی ضروری ہے۔
سب کو تعلیم کا حق ہونا چاہیے۔
            """.strip()
        }
        
        return sample_data.get(language, "Sample training data for multilingual tokenizer.")

    def train_tokenizer(self, training_file: str):
        """Train the SentencePiece tokenizer"""
        logger.info("Training SentencePiece tokenizer...")
        
        # Ensure model directory exists
        os.makedirs(settings.FINE_TUNED_MODEL_PATH, exist_ok=True)
        
        # Output paths
        model_prefix = os.path.join(settings.FINE_TUNED_MODEL_PATH, "multi_tokenizer")
        
        # Calculate adaptive vocabulary size based on training data
        # SentencePiece requires vocab_size <= unique_characters + some_buffer
        # For small datasets, we need to be more conservative
        adaptive_vocab_size = min(settings.SP_VOCAB_SIZE, max(1000, self.stats["total_sentences"] * 50))
        
        logger.info(f"Using adaptive vocabulary size: {adaptive_vocab_size} (based on {self.stats['total_sentences']} sentences)")
        
        # SentencePiece training arguments
        spm_args = [
            f'--input={training_file}',
            f'--model_prefix={model_prefix}',
            f'--vocab_size={adaptive_vocab_size}',
            f'--model_type={settings.SP_MODEL_TYPE}',
            f'--character_coverage={settings.SP_CHARACTER_COVERAGE}',
            f'--input_sentence_size={min(settings.SP_INPUT_SENTENCE_SIZE, self.stats["total_sentences"])}',
            '--shuffle_input_sentence=true' if settings.SP_SHUFFLE_INPUT_SENTENCE else '--shuffle_input_sentence=false',
            '--split_by_unicode_script=false',  # Don't split scripts to handle multilingual better
            '--split_by_whitespace=true',
            '--split_by_number=true',
            '--treat_whitespace_as_suffix=false',
            '--allow_whitespace_only_pieces=true',
            '--split_digits=false',  # Keep numbers intact
            '--byte_fallback=true',  # Handle unknown characters gracefully
            # Devanagari-specific settings - removed problematic NFC normalization
            '--remove_extra_whitespaces=true',
            '--add_dummy_prefix=false',  # Don't add dummy prefix for better multilingual support
        ]
        
        # Train the tokenizer
        try:
            spm.SentencePieceTrainer.train(' '.join(spm_args))
            logger.info(f"Tokenizer training completed successfully!")
            logger.info(f"Model saved as: {model_prefix}.model")
            logger.info(f"Vocabulary saved as: {model_prefix}.vocab")
            
            # Update settings paths
            self.update_settings_paths(f"{model_prefix}.model", f"{model_prefix}.vocab")
            
        except Exception as e:
            logger.error(f"Tokenizer training failed: {e}")
            raise

    def update_settings_paths(self, model_path: str, vocab_path: str):
        """Update settings.py with correct tokenizer paths"""
        try:
            settings_file = "core/settings.py"
            if os.path.exists(settings_file):
                with open(settings_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Update the tokenizer model path
                old_model_line = 'TOKENIZER_MODEL_PATH = "model/multi_tokenizer.model"'
                new_model_line = f'TOKENIZER_MODEL_PATH = "{model_path}"'
                content = content.replace(old_model_line, new_model_line)
                
                # Update the vocab path
                old_vocab_line = 'TOKENIZER_VOCAB_PATH = "model/multi_tokenizer.vocab"'
                new_vocab_line = f'TOKENIZER_VOCAB_PATH = "{vocab_path}"'
                content = content.replace(old_vocab_line, new_vocab_line)
                
                with open(settings_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info("Updated settings.py with new tokenizer paths")
        except Exception as e:
            logger.warning(f"Could not update settings.py: {e}")

    def test_tokenizer(self, model_path: str):
        """Test the trained tokenizer with sample texts"""
        logger.info("Testing trained tokenizer...")
        
        # Load the tokenizer
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        
        # Test sentences in different languages (representative sample from all 21 languages)
        test_sentences = {
            "assamese": "নমস্কাৰ, আপুনি কেনে আছে?",
            "bengali": "নমস্কার, আপনি কেমন আছেন?",
            "bodo": "नमस्कार, नों मिथिंनाय दं?",
            "english": "Hello, how are you today?",
            "gujurati": "નમસ્તે, તમે કેવા છો?",
            "hindi": "नमस्ते, आप कैसे हैं?",
            "kannada": "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?",
            "kashmiri": "نمسکار، تۄہہ کیتھ چھیو؟",
            "maithili": "नमस्कार, अहाँ केहन छी?",
            "malyalam": "നമസ്കാരം, നിങ്ങൾക്ക് എങ്ങനെയുണ്ട്?",
            "marathi": "नमस्कार, तुम्ही कसे आहात?",
            "meitei": "নমস্কার, নহাক কৈদোক্লেবগা?",
            "nepali": "नमस्कार, तपाईं कस्तो हुनुहुन्छ?",
            "odia": "ନମସ୍କାର, ଆପଣ କେମିତି ଅଛନ୍ତି?",
            "punjabi": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?",
            "sanskrit": "नमस्कार, कथं वर्तसे?",
            "santali": "ᱡᱚᱦᱟᱨ, ᱟᱢ ᱪᱮᱫᱮ ᱢᱮᱱᱟᱢᱟ?",
            "sindhi": "سلام، توهان ڪيئن آهيو؟",
            "tamil": "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
            "telugu": "నమస్కారం, మీరు ఎలా ఉన్నారు?",
            "urdu": "السلام علیکم، آپ کیسے ہیں؟"
        }
        
        logger.info("\nTokenization Test Results:")
        logger.info("=" * 50)
        
        for lang, sentence in test_sentences.items():
            tokens = sp.encode_as_pieces(sentence)
            ids = sp.encode_as_ids(sentence)
            reconstructed = sp.decode_pieces(tokens)
            
            logger.info(f"\n{lang.upper()}:")
            logger.info(f"Original: {sentence}")
            logger.info(f"Tokens: {tokens}")
            logger.info(f"Token count: {len(tokens)}")
            logger.info(f"Reconstructed: {reconstructed}")
            logger.info(f"Lossless: {'✓' if sentence == reconstructed else '✗'}")
        
        # Test vocabulary stats
        vocab_size = sp.get_piece_size()
        logger.info(f"\nVocabulary size: {vocab_size}")
        logger.info(f"Target vocabulary size: {settings.SP_VOCAB_SIZE}")
        
        # Test some special tokens
        logger.info("\nSpecial tokens:")
        logger.info(f"BOS: {sp.bos_id()} -> {sp.id_to_piece(sp.bos_id())}")
        logger.info(f"EOS: {sp.eos_id()} -> {sp.id_to_piece(sp.eos_id())}")
        logger.info(f"UNK: {sp.unk_id()} -> {sp.id_to_piece(sp.unk_id())}")
        
        # PAD token might not be defined in all SentencePiece models
        try:
            pad_id = sp.pad_id()
            if pad_id >= 0:  # Valid PAD ID
                logger.info(f"PAD: {pad_id} -> {sp.id_to_piece(pad_id)}")
            else:
                logger.info("PAD: Not defined in this model")
        except (IndexError, ValueError):
            logger.info("PAD: Not defined in this model")

    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_combined_file and os.path.exists(self.temp_combined_file):
            os.unlink(self.temp_combined_file)
            logger.info("Cleaned up temporary files")

    def print_statistics(self):
        """Print training data statistics"""
        logger.info("\nTraining Data Statistics:")
        logger.info("=" * 50)
        logger.info(f"Total sentences: {self.stats['total_sentences']}")
        logger.info(f"Devanagari sentences: {self.stats['devanagari_sentences']}")
        logger.info(f"Latin sentences: {self.stats['latin_sentences']}")
        logger.info(f"Mixed/Other sentences: {self.stats['total_sentences'] - self.stats['devanagari_sentences'] - self.stats['latin_sentences']}")
        
        logger.info("\nLanguage distribution:")
        for lang, count in self.stats['language_counts'].items():
            percentage = (count / self.stats['total_sentences']) * 100 if self.stats['total_sentences'] > 0 else 0
            logger.info(f"  {lang}: {count} sentences ({percentage:.1f}%)")


def main():
    """Main function to train the multilingual tokenizer"""
    logger.info("=" * 60)
    logger.info("Multilingual SentencePiece Tokenizer Training")
    logger.info("=" * 60)
    logger.info(f"Supported languages: {', '.join(settings.SUPPORTED_LANGUAGES)}")
    logger.info(f"Vocabulary size: {settings.SP_VOCAB_SIZE}")
    logger.info(f"Model type: {settings.SP_MODEL_TYPE}")
    logger.info(f"Character coverage: {settings.SP_CHARACTER_COVERAGE}")
    
    trainer = MultilingualTokenizerTrainer()
    
    try:
        # Step 1: Prepare training data
        training_file = trainer.prepare_training_data()
        trainer.print_statistics()
        
        # Step 2: Train tokenizer
        trainer.train_tokenizer(training_file)
        
        # Step 3: Test tokenizer
        model_path = os.path.join(settings.FINE_TUNED_MODEL_PATH, "multi_tokenizer.model")
        trainer.test_tokenizer(model_path)
        
        logger.info("\n" + "=" * 60)
        logger.info("Tokenizer training completed successfully!")
        logger.info("=" * 60)
        logger.info("Next steps:")
        logger.info("1. Restart your API server (app.py)")
        logger.info("2. Test the /tokenize endpoint with multilingual text")
        logger.info("3. Use the trained tokenizer for fine-tuning")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()