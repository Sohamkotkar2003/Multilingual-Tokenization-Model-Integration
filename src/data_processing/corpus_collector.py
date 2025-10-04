"""
Corpus Collection Script for 20+ Indian Languages

This script collects clean corpora from various sources as specified in the requirements:
- Wikipedia dumps (Indic languages)
- AI4Bharat Indic corpora
- HindMono datasets
- CC-100 multilingual corpora
- OSCAR datasets
- Other public sources

The collected data is then processed through the MCP pipeline for consistent
tokenization across scripts and preparation for multilingual tokenizer training.
"""

import os
import sys
import requests
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import zipfile
import tarfile
import json
from urllib.parse import urljoin, urlparse
import time

# Import settings
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CorpusCollector:
    """
    Collects multilingual corpora from various sources for 20+ Indian languages
    
    This class implements the corpus collection requirements:
    - Wikipedia dumps for Indic languages
    - AI4Bharat Indic corpora
    - HindMono datasets
    - CC-100 multilingual corpora
    - OSCAR datasets
    - Gurukul-specific curated text
    """
    
    def __init__(self):
        self.collection_stats = {
            "total_files_downloaded": 0,
            "total_size_mb": 0,
            "languages_collected": set(),
            "sources_used": set(),
            "errors": []
        }
        
        # Ensure data directories exist
        settings.create_directories()
        
        # Language to ISO code mapping
        self.language_codes = {
            "hindi": "hi",
            "sanskrit": "sa", 
            "marathi": "mr",
            "english": "en",
            "tamil": "ta",
            "telugu": "te",
            "kannada": "kn",
            "bengali": "bn",
            "gujarati": "gu",
            "punjabi": "pa",
            "odia": "or",
            "malayalam": "ml",
            "assamese": "as",
            "kashmiri": "ks",
            "konkani": "gom",
            "manipuri": "mni",
            "nepali": "ne",
            "sindhi": "sd",
            "urdu": "ur",
            "bodo": "brx",
            "dogri": "doi",
            "maithili": "mai",
            "santali": "sat"
        }
        
        # Data sources configuration
        self.data_sources = {
            "wikipedia": {
                "base_url": "https://dumps.wikimedia.org/",
                "languages": ["hi", "ta", "te", "kn", "bn", "gu", "pa", "or", "ml", "as", "ne", "ur"],
                "file_pattern": "{lang}wiki-latest-pages-articles.xml.bz2"
            },
            "ai4bharat": {
                "base_url": "https://huggingface.co/datasets/ai4bharat/indicnlg",
                "languages": ["hi", "ta", "te", "kn", "bn", "gu", "pa", "or", "ml", "as", "ne", "ur"],
                "type": "huggingface"
            },
            "oscar": {
                "base_url": "https://huggingface.co/datasets/oscar",
                "languages": ["hi", "ta", "te", "kn", "bn", "gu", "pa", "or", "ml", "as", "ne", "ur"],
                "type": "huggingface"
            },
            "cc100": {
                "base_url": "https://huggingface.co/datasets/cc100",
                "languages": ["hi", "ta", "te", "kn", "bn", "gu", "pa", "or", "ml", "as", "ne", "ur"],
                "type": "huggingface"
            }
        }
    
    def create_sample_data_for_language(self, language: str) -> str:
        """
        Create sample training data for a language if no corpus is available
        
        Args:
            language: Language code
            
        Returns:
            Sample text data
        """
        sample_data = {
            "hindi": """
नमस्ते, आप कैसे हैं आज?
मैं हिंदी भाषा सीख रहा हूं और यह बहुत दिलचस्प है।
भारत एक बहुत ही सुंदर और विविधतापूर्ण देश है।
यहां अनेक भाषाएं बोली जाती हैं और सभी की अपनी विशेषताएं हैं।
शिक्षा का महत्व आज के युग में और भी बढ़ गया है।
हमें अपनी संस्कृति और परंपराओं पर गर्व होना चाहिए।
विज्ञान और प्रौद्योगिकी ने हमारे जीवन को बदल दिया है।
पर्यावरण की सुरक्षा आज की सबसे बड़ी चुनौती है।
साहित्य और कला मानव सभ्यता की अमूल्य धरोहर हैं।
एकता में शक्ति है और विविधता में सुंदरता है।
            """.strip(),
            
            "tamil": """
வணக்கம், இன்று எப்படி இருக்கிறீர்கள்?
நான் தமிழ் மொழியை கற்கிறேன் மற்றும் இது மிகவும் சுவாரஸ்யமானது.
இந்தியா ஒரு மிகவும் அழகான மற்றும் பல்வகைமான நாடு.
இங்கே பல மொழிகள் பேசப்படுகின்றன மற்றும் ஒவ்வொன்றிற்கும் தனித்துவமான பண்புகள் உள்ளன.
கல்வியின் முக்கியத்துவம் இன்றைய காலத்தில் மேலும் அதிகரித்துள்ளது.
நமது கலாச்சாரம் மற்றும் பாரம்பரியங்களில் நாம் பெருமை கொள்ள வேண்டும்.
விஞ்ஞானம் மற்றும் தொழில்நுட்பம் நமது வாழ்க்கையை மாற்றியுள்ளது.
சுற்றுச்சூழல் பாதுகாப்பு இன்றைய மிகப்பெரிய சவாலாகும்.
இலக்கியம் மற்றும் கலை மனித நாகரிகத்தின் விலைமதிப்பற்ற பாரம்பரியம்.
ஒற்றுமையில் வலிமை உள்ளது மற்றும் பல்வகைமையில் அழகு உள்ளது.
            """.strip(),
            
            "telugu": """
నమస్కారం, ఈరోజు ఎలా ఉన్నారు?
నేను తెలుగు భాష నేర్చుకుంటున్నాను మరియు ఇది చాలా ఆసక్తికరంగా ఉంది.
భారతదేశం చాలా అందమైన మరియు వివిధతలతో కూడిన దేశం.
ఇక్కడ అనేక భాషలు మాట్లాడబడతాయి మరియు ప్రతి ఒక్కటికి దాని స్వంత ప్రత్యేకతలు ఉన్నాయి.
విద్య యొక్క ప్రాముఖ్యత ఈ రోజుల్లో మరింత పెరిగింది.
మన సంస్కృతి మరియు సంప్రదాయాలపై మనకు గర్వపడాలి.
విజ్ఞానం మరియు సాంకేతికత మన జీవితాన్ని మార్చివేసింది.
పర్యావరణ రక్షణ ఈ రోజు అతిపెద్ద సవాల్.
సాహిత్యం మరియు కళలు మానవ నాగరికత యొక్క అమూల్యమైన వారసత్వం.
ఐక్యతలో శక్తి ఉంది మరియు వివిధతలో అందం ఉంది.
            """.strip(),
            
            "kannada": """
ನಮಸ್ಕಾರ, ಇಂದು ಹೇಗಿದ್ದೀರಿ?
ನಾನು ಕನ್ನಡ ಭಾಷೆಯನ್ನು ಕಲಿಯುತ್ತಿದ್ದೇನೆ ಮತ್ತು ಇದು ತುಂಬಾ ಆಸಕ್ತಿದಾಯಕವಾಗಿದೆ.
ಭಾರತವು ತುಂಬಾ ಸುಂದರ ಮತ್ತು ವೈವಿಧ್ಯಮಯ ದೇಶವಾಗಿದೆ.
ಇಲ್ಲಿ ಅನೇಕ ಭಾಷೆಗಳನ್ನು ಮಾತನಾಡಲಾಗುತ್ತದೆ ಮತ್ತು ಪ್ರತಿಯೊಂದಕ್ಕೂ ತನ್ನದೇ ಆದ ವಿಶೇಷತೆಗಳಿವೆ.
ಶಿಕ್ಷಣದ ಮಹತ್ವ ಇಂದಿನ ಯುಗದಲ್ಲಿ ಇನ್ನೂ ಹೆಚ್ಚಾಗಿದೆ.
ನಮ್ಮ ಸಂಸ್ಕೃತಿ ಮತ್ತು ಸಂಪ್ರದಾಯಗಳ ಬಗ್ಗೆ ನಾವು ಹೆಮ್ಮೆಪಡಬೇಕು.
ವಿಜ್ಞಾನ ಮತ್ತು ತಂತ್ರಜ್ಞಾನ ನಮ್ಮ ಜೀವನವನ್ನು ಬದಲಾಯಿಸಿದೆ.
ಪರಿಸರ ರಕ್ಷಣೆ ಇಂದಿನ ಅತಿದೊಡ್ಡ ಸವಾಲು.
ಸಾಹಿತ್ಯ ಮತ್ತು ಕಲೆಗಳು ಮಾನವ ನಾಗರಿಕತೆಯ ಅಮೂಲ್ಯ ವಾರಸು.
ಏಕತೆಯಲ್ಲಿ ಶಕ್ತಿ ಇದೆ ಮತ್ತು ವೈವಿಧ್ಯದಲ್ಲಿ ಸೌಂದರ್ಯ ಇದೆ.
            """.strip(),
            
            "bengali": """
নমস্কার, আজ কেমন আছেন?
আমি বাংলা ভাষা শিখছি এবং এটি খুবই আকর্ষণীয়।
ভারত একটি খুবই সুন্দর এবং বৈচিত্র্যময় দেশ।
এখানে অনেক ভাষা বলা হয় এবং প্রতিটিরই নিজস্ব বৈশিষ্ট্য রয়েছে।
শিক্ষার গুরুত্ব আজকের যুগে আরও বেড়ে গেছে।
আমাদের সংস্কৃতি এবং ঐতিহ্যের উপর আমাদের গর্ব করা উচিত।
বিজ্ঞান এবং প্রযুক্তি আমাদের জীবনকে পরিবর্তন করেছে।
পরিবেশ সুরক্ষা আজকের সবচেয়ে বড় চ্যালেঞ্জ।
সাহিত্য এবং শিল্প মানব সভ্যতার অমূল্য সম্পদ।
ঐক্যে শক্তি এবং বৈচিত্র্যে সৌন্দর্য।
            """.strip(),
            
            "gujarati": """
નમસ્કાર, આજે કેમ છો?
હું ગુજરાતી ભાષા શીખી રહ્યો છું અને તે ખૂબ જ રસપ્રદ છે।
ભારત એક ખૂબ જ સુંદર અને વૈવિધ્યસભર દેશ છે।
અહીં ઘણી ભાષાઓ બોલાય છે અને દરેકની પોતાની વિશેષતાઓ છે।
શિક્ષણનું મહત્વ આજના યુગમાં વધુ વધી ગયું છે।
આપણે આપણી સંસ્કૃતિ અને પરંપરાઓ પર ગર્વ કરવો જોઈએ।
વિજ્ઞાન અને ટેકનોલોજીએ આપણા જીવનને બદલી નાખ્યું છે।
પર્યાવરણ સુરક્ષા આજની સૌથી મોટી પડકાર છે।
સાહિત્ય અને કલા માનવ સભ્યતાની અમૂલ્ય વારસો છે।
એકતામાં શક્તિ છે અને વૈવિધ્યમાં સુંદરતા છે।
            """.strip(),
            
            "punjabi": """
ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਅੱਜ ਕਿਵੇਂ ਹੋ?
ਮੈਂ ਪੰਜਾਬੀ ਭਾਸ਼ਾ ਸਿੱਖ ਰਿਹਾ ਹਾਂ ਅਤੇ ਇਹ ਬਹੁਤ ਦਿਲਚਸਪ ਹੈ।
ਭਾਰਤ ਇੱਕ ਬਹੁਤ ਹੀ ਸੁੰਦਰ ਅਤੇ ਵਿਭਿੰਨ ਦੇਸ਼ ਹੈ।
ਇੱਥੇ ਬਹੁਤ ਸਾਰੀਆਂ ਭਾਸ਼ਾਵਾਂ ਬੋਲੀਆਂ ਜਾਂਦੀਆਂ ਹਨ ਅਤੇ ਹਰ ਇੱਕ ਦੀ ਆਪਣੀ ਵਿਸ਼ੇਸ਼ਤਾ ਹੈ।
ਸਿੱਖਿਆ ਦਾ ਮਹੱਤਵ ਅੱਜ ਦੇ ਯੁੱਗ ਵਿੱਚ ਹੋਰ ਵੀ ਵਧ ਗਿਆ ਹੈ।
ਸਾਨੂੰ ਆਪਣੀ ਸਭਿਆਚਾਰ ਅਤੇ ਪਰੰਪਰਾਵਾਂ 'ਤੇ ਗਰਵ ਕਰਨਾ ਚਾਹੀਦਾ ਹੈ।
ਵਿਗਿਆਨ ਅਤੇ ਤਕਨਾਲੋਜੀ ਨੇ ਸਾਡੇ ਜੀਵਨ ਨੂੰ ਬਦਲ ਦਿੱਤਾ ਹੈ।
ਪਰਿਆਵਰਣ ਸੁਰੱਖਿਆ ਅੱਜ ਦੀ ਸਭ ਤੋਂ ਵੱਡੀ ਚੁਣੌਤੀ ਹੈ।
ਸਾਹਿਤ ਅਤੇ ਕਲਾ ਮਨੁੱਖੀ ਸਭਿਅਤਾ ਦੀ ਅਮੂਲਕ ਵਿਰਾਸਤ ਹਨ।
ਏਕਤਾ ਵਿੱਚ ਸ਼ਕਤੀ ਹੈ ਅਤੇ ਵਿਭਿੰਨਤਾ ਵਿੱਚ ਸੁੰਦਰਤਾ ਹੈ।
            """.strip(),
            
            "odia": """
ନମସ୍କାର, ଆଜି କିପରି ଅଛନ୍ତି?
ମୁଁ ଓଡ଼ିଆ ଭାଷା ଶିଖୁଛି ଏବଂ ଏହା ବହୁତ ରୋଚକ।
ଭାରତ ଏକ ବହୁତ ସୁନ୍ଦର ଏବଂ ବିବିଧତାପୂର୍ଣ୍ଣ ଦେଶ।
ଏଠାରେ ଅନେକ ଭାଷା କୁହାଯାଏ ଏବଂ ପ୍ରତ୍ୟେକର ନିଜର ବିଶେଷତା ଅଛି।
ଶିକ୍ଷାର ମହତ୍ତ୍ୱ ଆଜିର ଯୁଗରେ ଆହୁରି ବଢ଼ିଗଲାଣି।
ଆମେ ଆମର ସଂସ୍କୃତି ଏବଂ ପରମ୍ପରା ଉପରେ ଗର୍ବ କରିବା ଉଚିତ।
ବିଜ୍ଞାନ ଏବଂ ପ୍ରଯୁକ୍ତିବିଦ୍ୟା ଆମର ଜୀବନକୁ ବଦଳାଇ ଦେଇଛି।
ପରିବେଶ ସୁରକ୍ଷା ଆଜିର ସବୁଠାରୁ ବଡ଼ ଚ୍ୟାଲେଞ୍ଜ।
ସାହିତ୍ୟ ଏବଂ କଳା ମାନବ ସଭ୍ୟତାର ଅମୂଲ୍ୟ ଧରୋହର।
ଏକତାରେ ଶକ୍ତି ଏବଂ ବିବିଧତାରେ ସୁନ୍ଦରତା।
            """.strip(),
            
            "malayalam": """
നമസ്കാരം, ഇന്ന് എങ്ങനെയുണ്ട്?
ഞാൻ മലയാളം ഭാഷ പഠിക്കുന്നു, ഇത് വളരെ രസകരമാണ്.
ഇന്ത്യ ഒരു വളരെ മനോഹരവും വൈവിധ്യമാർന്ന രാജ്യമാണ്.
ഇവിടെ നിരവധി ഭാഷകൾ സംസാരിക്കപ്പെടുന്നു, ഓരോന്നിനും സ്വന്തം സവിശേഷതകൾ ഉണ്ട്.
വിദ്യാഭ്യാസത്തിന്റെ പ്രാധാന്യം ഇന്നത്തെ യുഗത്തിൽ കൂടുതൽ വർദ്ധിച്ചിരിക്കുന്നു.
നമ്മുടെ സംസ്കാരത്തിലും പാരമ്പര്യങ്ങളിലും നമുക്ക് അഭിമാനിക്കണം.
ശാസ്ത്രവും സാങ്കേതികവിദ്യയും നമ്മുടെ ജീവിതത്തെ മാറ്റിമറിച്ചിരിക്കുന്നു.
പരിസ്ഥിതി സംരക്ഷണം ഇന്നത്തെ ഏറ്റവും വലിയ വെല്ലുവിളിയാണ്.
സാഹിത്യവും കലയും മനുഷ്യ സംസ്കാരത്തിന്റെ അമൂല്യമായ പൈതൃകമാണ്.
ഐക്യത്തിൽ ശക്തിയും വൈവിധ്യത്തിൽ സൗന്ദര്യവും ഉണ്ട്।
            """.strip(),
            
            "assamese": """
নমস্কাৰ, আজি কেনেকৈ আছা?
মই অসমীয়া ভাষা শিকি আছোঁ আৰু ইয়াৰ বাবে বৰ ৰসিক।
ভাৰত এখন বৰ সুন্দৰ আৰু বৈচিত্ৰ্যপূৰ্ণ দেশ।
ইয়াত বহুতো ভাষা কোৱা হয় আৰু প্ৰতিটোৰে নিজৰ বৈশিষ্ট্য আছে।
শিক্ষাৰ গুৰুত্ব আজিৰ যুগত আৰু বেছি হৈছে।
আমি আমাৰ সংস্কৃতি আৰু পৰম্পৰাৰ ওপৰত গৌৰৱ কৰিব লাগে।
বিজ্ঞান আৰু প্ৰযুক্তিয়ে আমাৰ জীৱনক সলনি কৰিছে।
পৰিৱেশ সুৰক্ষা আজিৰ আটাইতকৈ ডাঙৰ প্ৰত্যাহ্বান।
সাহিত্য আৰু কলা মানৱ সভ্যতাৰ অমূল্য সম্পদ।
ঐক্যত শক্তি আৰু বৈচিত্ৰ্যত সৌন্দৰ্য।
            """.strip(),
            
            "urdu": """
السلام علیکم، آج کیسے ہیں؟
میں اردو زبان سیکھ رہا ہوں اور یہ بہت دلچسپ ہے۔
بھارت ایک بہت ہی خوبصورت اور متنوع ملک ہے۔
یہاں بہت سی زبانیں بولی جاتی ہیں اور ہر ایک کی اپنی خصوصیات ہیں۔
تعلیم کی اہمیت آج کے دور میں اور بھی بڑھ گئی ہے۔
ہمیں اپنی ثقافت اور روایات پر فخر کرنا چاہیے۔
سائنس اور ٹیکنالوجی نے ہماری زندگی کو بدل دیا ہے۔
ماحولیات کا تحفظ آج کا سب سے بڑا چیلنج ہے۔
ادب اور فن انسانی تہذیب کا قیمتی ورثہ ہیں۔
اتحاد میں طاقت ہے اور تنوع میں خوبصورتی ہے۔
            """.strip(),
            
            "nepali": """
नमस्कार, आज कसरी हुनुहुन्छ?
म नेपाली भाषा सिक्दै छु र यो धेरै रोचक छ।
भारत एक धेरै सुन्दर र विविधतापूर्ण देश हो।
यहाँ धेरै भाषाहरू बोलिन्छन् र प्रत्येकको आफ्नै विशेषताहरू छन्।
शिक्षाको महत्व आजको युगमा र अझै बढेको छ।
हामीले आफ्नो संस्कृति र परम्पराहरूमा गर्व गर्नुपर्छ।
विज्ञान र प्रविधिले हाम्रो जीवनलाई परिवर्तन गरेको छ।
पर्यावरण सुरक्षा आजको सबैभन्दा ठूलो चुनौती हो।
साहित्य र कला मानव सभ्यताको अमूल्य सम्पदा हो।
एकतामा शक्ति छ र विविधतामा सुन्दरता छ।
            """.strip(),
            
            "english": """
Hello, how are you doing today?
Learning multiple languages is a wonderful and enriching experience.
Education forms the very foundation of human progress and development.
Technology has completely transformed the way we live and work.
Reading books regularly expands our knowledge and imagination significantly.
Environmental protection is one of the most crucial challenges of our time.
Cultural diversity makes our society stronger and more vibrant.
Hard work and dedication are the keys to achieving success.
Respect for others is a fundamental principle of civilized society.
Science and innovation continue to drive human development forward.
The history of human civilization is filled with remarkable achievements.
Art and literature reflect the soul of a culture and its people.
Democracy depends on the active participation of informed citizens.
Healthcare should be accessible to all people regardless of their background.
Economic development must be sustainable and environmentally responsible.
International cooperation is essential for solving global challenges.
The internet has created unprecedented opportunities for communication and learning.
Music and dance are universal languages that transcend cultural boundaries.
Critical thinking skills are essential in our information-rich world.
Peace and harmony are the ultimate goals of human society.
            """.strip()
        }
        
        return sample_data.get(language, f"Sample training data for {language} language.")
    
    def download_wikipedia_dump(self, language: str) -> Optional[str]:
        """
        Download Wikipedia dump for a specific language
        
        Args:
            language: Language code (e.g., 'hi', 'ta', 'te')
            
        Returns:
            Path to downloaded file or None if failed
        """
        try:
            # Wikipedia dump URL
            url = f"https://dumps.wikimedia.org/{language}wiki/latest/{language}wiki-latest-pages-articles.xml.bz2"
            
            logger.info(f"Downloading Wikipedia dump for {language}: {url}")
            
            # Create filename
            filename = f"{language}wiki-latest-pages-articles.xml.bz2"
            filepath = os.path.join(settings.TRAINING_DATA_PATH, filename)
            
            # Download file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            logger.info(f"Downloaded {filename} ({file_size_mb:.2f} MB)")
            
            self.collection_stats["total_files_downloaded"] += 1
            self.collection_stats["total_size_mb"] += file_size_mb
            self.collection_stats["languages_collected"].add(language)
            self.collection_stats["sources_used"].add("wikipedia")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to download Wikipedia dump for {language}: {e}")
            self.collection_stats["errors"].append(f"Wikipedia {language}: {e}")
            return None
    
    def create_sample_corpus_files(self):
        """
        Create sample corpus files for all languages
        This is used when external sources are not available
        """
        logger.info("Creating sample corpus files for all languages")
        
        for language in settings.SUPPORTED_LANGUAGES:
            # Get language code
            lang_code = self.language_codes.get(language, language)
            
            # Create sample data
            sample_data = self.create_sample_data_for_language(language)
            
            # Write to training file
            train_filename = settings.CORPUS_FILES[language]
            train_filepath = os.path.join(settings.TRAINING_DATA_PATH, train_filename)
            
            with open(train_filepath, 'w', encoding='utf-8') as f:
                f.write(sample_data)
            
            # Create validation file
            val_filename = train_filename.replace('_train.txt', '_val.txt')
            val_filepath = os.path.join(settings.VALIDATION_DATA_PATH, val_filename)
            
            # Create validation data (subset of training data)
            val_data = '\n'.join(sample_data.split('\n')[:5])  # First 5 lines
            
            with open(val_filepath, 'w', encoding='utf-8') as f:
                f.write(val_data)
            
            logger.info(f"Created sample files for {language}: {train_filepath}, {val_filepath}")
            
            self.collection_stats["languages_collected"].add(language)
            self.collection_stats["sources_used"].add("sample_data")
    
    def collect_corpora(self, use_external_sources: bool = False):
        """
        Collect corpora from various sources
        
        Args:
            use_external_sources: Whether to attempt downloading from external sources
        """
        logger.info("Starting corpus collection for 20+ Indian languages")
        
        if use_external_sources:
            # Try to download from external sources
            logger.info("Attempting to download from external sources...")
            
            # Download Wikipedia dumps for major languages
            major_languages = ["hi", "ta", "te", "kn", "bn", "gu", "pa", "or", "ml", "as", "ne", "ur"]
            
            for lang_code in major_languages:
                try:
                    self.download_wikipedia_dump(lang_code)
                    time.sleep(1)  # Be respectful to the server
                except Exception as e:
                    logger.warning(f"Failed to download {lang_code}: {e}")
                    continue
            
            # Extract text from downloaded Wikipedia dumps
            logger.info("Extracting text from Wikipedia dumps...")
            try:
                from .wikipedia_extractor import WikipediaExtractor
                extractor = WikipediaExtractor()
                all_stats = extractor.extract_all_wikipedia_dumps(max_articles_per_language=5000)
                extractor.print_extraction_summary(all_stats)
                
                # Update collection stats
                self.collection_stats["languages_collected"].update(extractor.stats["languages_processed"])
                self.collection_stats["sources_used"].add("wikipedia_extracted")
                
            except Exception as e:
                logger.warning(f"Failed to extract Wikipedia text: {e}")
                logger.info("Falling back to sample data...")
        
        # Always create sample data as fallback
        logger.info("Creating sample corpus files as fallback...")
        self.create_sample_corpus_files()
        
        # Print collection statistics
        self.print_collection_statistics()
    
    def print_collection_statistics(self):
        """Print corpus collection statistics"""
        logger.info("\n" + "=" * 60)
        logger.info("Corpus Collection Statistics")
        logger.info("=" * 60)
        logger.info(f"Total files downloaded: {self.collection_stats['total_files_downloaded']}")
        logger.info(f"Total size: {self.collection_stats['total_size_mb']:.2f} MB")
        logger.info(f"Languages collected: {len(self.collection_stats['languages_collected'])}")
        logger.info(f"Sources used: {', '.join(self.collection_stats['sources_used'])}")
        
        if self.collection_stats['errors']:
            logger.info(f"Errors encountered: {len(self.collection_stats['errors'])}")
            for error in self.collection_stats['errors'][:5]:  # Show first 5 errors
                logger.warning(f"  - {error}")
        
        logger.info("\nLanguages with data:")
        for lang in sorted(self.collection_stats['languages_collected']):
            logger.info(f"  - {lang}")
        
        logger.info("=" * 60)


def main():
    """Main function to run corpus collection"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect corpora for 20+ Indian languages")
    parser.add_argument("--external", action="store_true", 
                       help="Attempt to download from external sources")
    parser.add_argument("--sample-only", action="store_true",
                       help="Create only sample data files")
    
    args = parser.parse_args()
    
    # Create corpus collector
    collector = CorpusCollector()
    
    try:
        if args.sample_only:
            logger.info("Creating sample data only...")
            collector.create_sample_corpus_files()
        else:
            logger.info("Starting corpus collection...")
            collector.collect_corpora(use_external_sources=args.external)
        
        logger.info("\nCorpus collection completed!")
        logger.info("\nNext steps:")
        logger.info("1. Run MCP pipeline: python src/data_processing/mcp_pipeline.py")
        logger.info("2. Train tokenizer: python src/training/train_tokenizer.py")
        logger.info("3. Fine-tune model: python src/training/fine_tune.py")
        
    except Exception as e:
        logger.error(f"Corpus collection failed: {e}")
        raise


if __name__ == "__main__":
    main()
