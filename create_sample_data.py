"""
Sample Training Data Generator

This script creates sample training and validation data files for all supported languages.
Run this before training the tokenizer if you don't have your own corpus files.

Usage:
    python create_sample_data.py
"""

import os
from core import settings

def create_sample_training_data():
    """Create comprehensive sample training data for all languages"""
    
    # Ensure directories exist
    settings.create_directories()
    
    # Hindi training data
    hindi_training = """नमस्ते, आप कैसे हैं आज?
मैं हिंदी भाषा सीख रहा हूं और यह बहुत दिलचस्प है।
भारत एक बहुत ही सुंदर और विविधतापूर्ण देश है।
यहां अनेक भाषाएं बोली जाती हैं और सभी की अपनी विशेषताएं हैं।
शिक्षा का महत्व आज के युग में और भी बढ़ गया है।
हमें अपनी संस्कृति और परंपराओं पर गर्व होना चाहिए।
विज्ञान और प्रौद्योगिकी ने हमारे जीवन को बदल दिया है।
पर्यावरण की सुरक्षा आज की सबसे बड़ी चुनौती है।
साहित्य और कला मानव सभ्यता की अमूल्य धरोहर हैं।
एकता में शक्ति है और विविधता में सुंदरता है।
स्वतंत्रता संग्राम में अनेक वीरों ने अपना बलिदान दिया था।
गांधी जी ने अहिंसा और सत्याग्रह का मार्ग दिखाया था।
योग और आयुर्वेद भारत की प्राचीन विद्याएं हैं।
त्योहार हमारे जीवन में खुशियां और रंग भरते हैं।
संगीत और नृत्य कला के सुंदर रूप हैं।
गणित और विज्ञान की शिक्षा बहुत महत्वपूर्ण है।
कृषि भारत की अर्थव्यवस्था का आधार है।
नदियां हमारे देश की जीवनरेखा हैं और उनका संरक्षण जरूरी है।
बच्चों की शिक्षा में माता-पिता की भूमिका अत्यंत महत्वपूर्ण है।
अच्छे स्वास्थ्य के लिए संतुलित आहार और व्यायाम आवश्यक है।"""

    # Sanskrit training data
    sanskrit_training = """नमस्कारः, भवान् कथं वर्तते अद्य?
संस्कृतं भारतस्य प्राचीनतमा भाषा अस्ति।
वेदाः संस्कृते लिखिताः सन्ति और ये ज्ञान के भंडार हैं।
धर्मो रक्षति रक्षितः इति उक्तम् अस्ति।
सत्यं शिवं सुन्दरम् इति त्रिगुण ब्रह्मस्य।
विद्या ददाति विनयं विनयाद् याति पात्रताम्।
यत्र नार्यस्तु पूज्यन्ते रमन्ते तत्र देवताः।
सर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः।
वसुधैव कुटुम्बकम् इति महान् सिद्धान्तः।
अहिंसा परमो धर्मः इति भारतीय चिन्तनम्।
गुरुर्ब्रह्मा गुरुर्विष्णुः गुरुर्देवो महेश्वरः।
माता च पिता च प्रथमे गुरुः भवतः।
ॐ शान्तिः शान्तिः शान्तिः इति मन्त्रः।
कर्मण्येवाधिकारस्ते मा फलेषु कदाचन।
श्लोकाः संस्कृत साहित्यस्य सुन्दर रूपाणि सन्ति।
उपनिषदाः दर्शनस्य गहन ग्रन्थाः सन्ति।
रामायणं महाकाव्यम् अस्ति।
महाभारतं इतिहासस्य महान् ग्रन्थः अस्ति।
पुराणानि धर्म और संस्कृति के स्रोत हैं।
योगः मनः और शरीरस्य एकत्वम् अस्ति।
आयुर्वेदः प्राकृतिक चिकित्सा पद्धतिः अस्ति।
संस्कृत व्याकरणं अत्यन्त वैज्ञानिकम् अस्ति।"""

    # Marathi training data
    marathi_training = """नमस्कार, तुम्ही कसे आहात आज?
मराठी ही महाराष्ट्राची गौरवशाली भाषा आहे.
मला मराठी भाषा खूप आवडते आणि ती शिकायला मजा येते.
शिक्षण हा आपल्या जीवनातील सर्वात महत्वाचा भाग आहे.
पुस्तक वाचणे हा एक अतिशय चांगला सवय आहे.
निसर्गाचे संरक्षण करणे आपले कर्तव्य आहे.
सणउत्सव आपल्या संस्कृतीचा महत्वाचा भाग आहेत.
कष्ट केल्याशिवाय जीवनात काहीही मिळत नाही.
एकता आणि अखंडता हीच खरी शक्ती आहे.
सत्य आणि न्याय हेच जीवनाचे खरे मूल्य आहेत.
छत्रपती शिवाजी महाराज हे आमचे आदर्श आहेत.
संत तुकाराम आणि संत ज्ञानेश्वर हे मराठी साहित्याचे रत्न आहेत.
लावणी आणि तमाशा हे आमचे लोकनृत्य आहेत.
पुणे आणि मुंबई ही महाराष्ट्राची महत्वाची शहरे आहेत.
कोकण, विदर्भ, मराठवाडा हे वेगवेगळे प्रदेश आहेत.
वडापाव आणि पावभाजी हे आमचे स्थानिक खाद्यपदार्थ आहेत.
गणपती बाप्पा मोरया हा आमचा प्रिय जयघोष आहे.
शिक्षण, आरोग्य आणि न्याय यावर भर द्यायला हवा.
कृषी हाच आमच्या राज्याचा मुख्य व्यवसाय आहे.
मराठी भाषेला जागतिक स्तरावर पोहोचवायचे आहे."""

    # English training data
    english_training = """Hello, how are you doing today?
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
Peace and harmony are the ultimate goals of human society."""

    # Create training files
    training_files = [
        ("hi_train.txt", hindi_training),
        ("sa_train.txt", sanskrit_training),
        ("mr_train.txt", marathi_training),
        ("en_train.txt", english_training)
    ]
    
    for filename, content in training_files:
        filepath = os.path.join(settings.TRAINING_DATA_PATH, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created training file: {filepath}")

def create_sample_validation_data():
    """Create validation data for testing"""
    
    # Hindi validation data
    hindi_validation = """आज मौसम कैसा है?
क्या आप मेरी सहायता कर सकते हैं?
भारतीय संस्कृति बहुत समृद्ध है।
गणित एक महत्वपूर्ण विषय है।
हमें प्रकृति की देखभाल करनी चाहिए।"""

    # Sanskrit validation data
    sanskrit_validation = """अद्य वातावरणं कीदृशम् अस्ति?
किं भवान् मम सहायतां कर्तुं शक्नोति?
भारतीयः संस्कृतिः अत्यन्त समृद्धः अस्ति।
गणितं महत्वपूर्णं विषयम् अस्ति।
प्रकृत्याः रक्षणं कर्तव्यम् अस्ति।"""

    # Marathi validation data
    marathi_validation = """आज हवामान कसे आहे?
तुम्ही माझी मदत करू शकता का?
भारतीय संस्कृती खूप समृद्ध आहे.
गणित हा महत्वाचा विषय आहे.
निसर्गाची काळजी घेणे गरजेचे आहे."""

    # English validation data
    english_validation = """What is the weather like today?
Can you help me with this problem?
Indian culture is very rich and diverse.
Mathematics is an important subject to study.
We need to take care of our environment."""

    validation_files = [
        ("hi_val.txt", hindi_validation),
        ("sa_val.txt", sanskrit_validation), 
        ("mr_val.txt", marathi_validation),
        ("en_val.txt", english_validation)
    ]
    
    for filename, content in validation_files:
        filepath = os.path.join(settings.VALIDATION_DATA_PATH, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created validation file: {filepath}")

def main():
    """Create all sample data files"""
    print("Creating sample training and validation data...")
    print("=" * 50)
    
    create_sample_training_data()
    print()
    create_sample_validation_data()
    
    print("=" * 50)
    print("Sample data creation completed!")
    print()
    print("Next steps:")
    print("1. Run: python train_tokenizer.py")
    print("2. Run: python train.py") 
    print("3. Start the API: python app.py")
    print()
    print("Note: Replace these sample files with your own")
    print("larger training corpus for better results.")

if __name__ == "__main__":
    main()