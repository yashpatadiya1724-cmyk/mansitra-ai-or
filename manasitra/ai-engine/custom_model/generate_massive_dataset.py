import json
import random
import os
import itertools

DATASET_PATH = "psychology_dataset.json"

CATEGORIES = [
    "Exam stress", "Breakup", "Anxiety", "Loneliness", 
    "Family pressure", "Career confusion", "Motivation", "Self-esteem"
]

# --- TEMPLATE POOLS ---

# 1. EXAM STRESS
EXAM_USER_EN = [
    "I have exams coming up and I am totally freaking out.",
    "My finals are next week and I can't concentrate at all.",
    "I feel like I'm going to fail my exams no matter how much I study."
]
EXAM_USER_HI = [
    "Bhai exam aa rahi hai aur meri bohot fati padi hai.",
    "Padhai me man hi nahi lag raha, lag raha hai sab bhul gaya hu.",
    "Exams ka itna tension hai ki raat ko nind nahi aati."
]
EXAM_USER_GU = [
    "Mari exam aavi rahi che ne mane bau tension thay che.",
    "Hu jetlu pan vachu, exam na nam thi j badhu bhulai jaay che.",
    "Exam nu aatlu badhu pressure che ke mane khabar nathi padti shu karu."
]
EXAM_RESP_EN = [
    "It sounds like the pressure of finals is really weighing heavily on you. What specifically feels the most overwhelming right now?",
    "I hear how stressed you are about failing despite studying. Let's take a breath. How are you breaking down your study sessions?",
    "It makes complete sense that you're freaking out when exams are this close. What is one small topic you feel confident about?"
]
EXAM_RESP_HI = [
    "Mai samajh sakta hu ki exam ka itna heavy pressure kaisa lagta hai. Ye darr natural hai. Abhi sabse zyada tension kis subject ka ho raha hai?",
    "Aapko lag raha hai ki aap sab bhul gaye, jo ki stress me aam baat hai. Kya hum milkar ek chhota schedule bana sakte hai?",
    "Nind na aana exam anxiety ka bada symptom hai. Aapke mind me is waqt sabse zyada kya chal raha hai?"
]
EXAM_RESP_GU = [
    "Exam nu tension aavvu aam vaat che, ane tamaro ghabrahat hu samji shaku chu. Tame atyare sauthi vadhu chinta kya vishay ni kari rahya cho?",
    "Bhulai javu ae pressure nu parinam che. Tame shaanti thi vicharo, shu aapne ek nano study plan banavi shakiye?",
    "Aatlu badhu pressure sahvu agahru che. Tame khud ne thodo aaram aapvanu vicharyu che?"
]

# 2. BREAKUP
BRK_USER_EN = [
    "We just broke up and I feel like my chest is completely empty.",
    "I can't stop checking their social media. It hurts so much.",
    "I gave everything to this relationship and now I'm left with nothing."
]
BRK_USER_HI = [
    "Mera breakup ho gaya aur samajh nahi aa raha kaise move on karu.",
    "Mai baar baar usko online check karta hu, bohot dard hota hai.",
    "Maine apna sab kuch usko de diya tha, ab main bilkul akela pad gaya hu."
]
BRK_USER_GU = [
    "Amaru breakup thai gayu che ane mane andar thi ekdam khali laagya kare che.",
    "Hu vaare vaare aeno photo jovu chu, bau dukh thay che mane.",
    "Me aa relation ma maru badhu aapi didhu hatu, have mari pase kai nathi."
]
BRK_RESP_EN = [
    "I can hear how incredibly painful this emptiness feels for you right now. Healing takes time, but what is one small thing bringing you comfort today?",
    "It is completely natural to want to check their social media when you're grieving a loss. What do you feel right after you look at their profile?",
    "Giving your all and feeling left with nothing is a very heavy burden. Let's acknowledge your capacity to love. How are you taking care of yourself today?"
]
BRK_RESP_HI = [
    "Breakup ka dard sach me bohot gehra hota hai, aur move on karna aasan nahi. Aapko is waqt kis baat ka sabse zyada darr lag raha hai?",
    "Online check karna aapke dil ki aadat ban chuki hai, aur use todna mushkil hai. Kya aap is dukh ko likhne ka try kar sakte hai?",
    "Jab hum apna sab kuch dete hai aur badle me akelapan milta hai, to bohot takleef hoti hai. Aap apni iss takleef ko kaise sambhal rahe hai?"
]
BRK_RESP_GU = [
    "Breakup pachi andar thi khali panu laagvu ae tamara saacha prem ni nishani che. Aa dukh maa tame potani sambhal kevi rite rakhi rahya cho?",
    "Photo jova thi dukh thavu swabhavik che. Shu tame aapna dhyan ne thodi vaar mate biji koi jagya par dori shako cho?",
    "Tamari pase kai nathi aevu lagvu bau pidadayak che. Pan tamari andar ni takat aje pan kachbandh che. Tame atyare shu vichari rahya cho?"
]

# 3. ANXIETY
ANX_USER_EN = [
    "My chest is tight and I keep overthinking every little detail.",
    "I feel this constant sense of dread and I don't even know why.",
    "I am having a panic attack, my breathing is so fast right now."
]
ANX_USER_HI = [
    "Mera dil bohot ghabra raha hai aur overthinking ruk nahi rahi.",
    "Ek ajeeb sa darr lagta rehta hai poora din, wajah bhi nahi pata.",
    "Mujhe panic attack jaisa feel ho raha hai, saans tez chal rahi hai."
]
ANX_USER_GU = [
    "Mari chhati ma bhar laagya kare che ane hu satat overthink karu chu.",
    "Mane aakho divas ek ajeeb darr lagya kare che, khabar nai kem.",
    "Mane ghabraman thai rahi che, mari shwaas bau jadap thi chaali rahi che."
]
ANX_RESP_EN = [
    "I hear how overwhelming that chest tightness and overthinking must be. Can you try focusing on five things you can see around you right now?",
    "A constant sense of dread without a clear reason is exhausting. How does this dread physically feel in your body?",
    "I am here with you. Please take a slow, deep breath in with me. What is the physical sensation you're noticing the most right now?"
]
ANX_RESP_HI = [
    "Overthinking aur ghabrahat sach me bohot thaka dene wali hoti hai. Kya aap ek lambi saans lekar mujhe aas paas ki 3 cheezein bata sakte hai?",
    "Bina wajah ka darr sabse zyada bechain karta hai. Ye darr aapke din ke kis hisse me sabse zyada badhta hai?",
    "Mai aapke sath hu, aap akele nahi hai. Dheere dheere saans lijiye. Aapko apni body me abhi kaisa feel ho raha hai?"
]
ANX_RESP_GU = [
    "Overthinking ane chhati ma bhar lagvu ae anxiety na lakshano che. Shu tame mara sathe ek lambo shwas lai shako cho?",
    "Karan vagar no darr mansik rite thakvadi de che. Tame aa darr thi bachva mate shu karo cho?",
    "Hu tamari sathe j chu. Ghabravani jarur nathi. Atyare tamne sharir maa kya sauthi vadhu bechaini laage che?"
]

# 4. LONELINESS
LON_USER_EN = [
    "I am surrounded by people but I feel completely invisible.",
    "I have nobody to text or talk to when I get home.",
    "Weekends are the hardest because I realize how alone I actually am."
]
LON_USER_HI = [
    "Bheed me bhi mai khud ko bilkul akela mehsus karta hu.",
    "Ghar aane ke baad kisi se baat karne ko nahi hota, sannata chubhta hai.",
    "Weekends sabse zyada dard dete hai kyunki tab mera akelapan saamne aa jata hai."
]
LON_USER_GU = [
    "Ghana loko vachhe hova chata hu ekdum akelo anubhavu chu.",
    "Ghare aavya pachi vaat karva mate koi nathi hotu.",
    "Raja na divaso maa mane maro ekalvayo panu bau j satave che."
]
LON_RESP_EN = [
    "Feeling invisible while surrounded by people is a very specific, deep kind of pain. What do you wish someone would notice about you?",
    "That silence when you get home can definitely feel heavy. What is one small activity that brings you a tiny bit of comfort in those moments?",
    "It sounds like weekends magnify the isolation. If you could build an ideal, gentle weekend for yourself, what would it look like?"
]
LON_RESP_HI = [
    "Bheed me akela mehsus karna sach me dil ko todne wala hota hai. Aap kaisa connection dhundh rahe hai iss waqt?",
    "Ghar ka sannata jab chubhne lage, tab khud se judna zaroori ho jata hai. Kya aapko koi aisi cheez pasand hai jisme aap khud ko busy rakh sake?",
    "Weekends ka khali pan aksar akelepan ko bada deta hai. Kya hum koi chhota routine soch sakte hai jo aapke weekend ko thoda behtar banaye?"
]
LON_RESP_GU = [
    "Loko vachhe akela lagvu ae antar-man ni vyatha che. Tame atyare keva vyakti ni saathe vaat karvani iccha rakho cho?",
    "Ghar ni shanti jyaare kachdi nakhe, tyare aapne potani jaat sathe vaat karvi joiye. Tamne ekla kyaare gamtu hoy che?",
    "Raja na divaso maa aa ekalpanu vadhu pidayadayak hoy che. Shu tame tami jaat ne thodo samay aapi shako cho?"
]

# 5. FAMILY PRESSURE
FAM_USER_EN = [
    "My parents expect me to be perfect and it's slowly crushing me.",
    "They constantly compare me to my cousins. I feel like a disappointment.",
    "I want to choose a different career, but my family won't even listen."
]
FAM_USER_HI = [
    "Mere parents ko har cheez perfect chahiye aur ye pressure mujhe daba raha hai.",
    "Wo hamesha mujhe mere cousins se compare karte hai, mujhe lagta hai mai failure hu.",
    "Mai alag career chahta hu par meri family meri baat sunne ko hi taiyar nahi hai."
]
FAM_USER_GU = [
    "Mara mata-pita mane perfect jova mange che ane aa pressure mane dabaavi rahyu che.",
    "Teo satat mari sarkhamani mara cousin sathe kare che. Mane maari jaat par gusso aave che.",
    "Mane aek alag career pasand karvu che pan maro parivar mari vaat sambhalto nathi."
]
FAM_RESP_EN = [
    "Living under the weight of perfectionism sounds absolutely crushing. How do you separate your own desires from their expectations?",
    "Constant comparison is deeply unfair and painful. I hear how much it hurts to feel like a disappointment. What are you proud of in yourself?",
    "It must be incredibly frustrating to not feel heard about your own life choices. What draws you to this different career path?"
]
FAM_RESP_HI = [
    "Perfection ki ummeed ke bojh tale jeena bohot thaka dene wala hai. Kya aap khud ki nazron me apni mehnat ko dekh paate hai?",
    "Bar bar kisi aur se comparison hona bohot dard deta hai. Mai sun raha hu aapki takleef. Aap khud me kya achhai dekhte hai?",
    "Apni zindagi ka faisla na le pana bohot nirasha janak hai. Aapko us naye career me kya pasand hai jo family ko nahi dikh raha?"
]
FAM_RESP_GU = [
    "Perfect banvanu pressure mansik rite todi nakhe che. Shu tame tamari potani safalta ne mahatva aapo cho?",
    "Satat sarkhamani thava thi confidence ghati jaay che. Pan hu tamari mehnat samji shaku chu. Tame tamari jaat ma kai saari vaat joi che?",
    "Tamari vaat koi na sambhale tyare gusso aavvo aam che. Tame je career pasand karyu che, tema tamne shu game che?"
]

# 6. CAREER CONFUSION
CAR_USER_EN = [
    "I feel like everyone is moving ahead in their career except me.",
    "I hate my current job but I have no idea what else I can do.",
    "I am constantly doubting if I chose the right path. It's paralyzing."
]
CAR_USER_HI = [
    "Mujhe lagta hai sab life me aage badh gaye hai, sirf mai hi atka hua hu.",
    "Mujhe apni job se nafrat hai par samajh nahi aata ki dusra kya karu.",
    "Mujhe hamesha doubt rehta hai ki maine sahi raasta chuna hai ya nahi."
]
CAR_USER_GU = [
    "Mane laage che ke badha aagal vadhi gaya che, pan hu ahiya j atkayelo chu.",
    "Mane mari job jara pan nathi gamti, pan biju shu karu te samjatu nathi.",
    "Mane satat sanka thaya kare che ke me saacho rasto pasand karyo che ke nahi."
]
CAR_RESP_EN = [
    "Watching others move ahead while feeling stuck is a very isolating experience. What does 'moving ahead' actually look like for you, personally?",
    "It sounds exhausting to wake up to a job you hate without a clear exit plan. What did you enjoy doing before this job?",
    "Self-doubt can be absolutely paralyzing. If you strip away all external expectations, what does your gut tell you?"
]
CAR_RESP_HI = [
    "Dusro ko aage badhte dekh kar atka hua mehsoos karna bohot frustrating hota hai. Aapke hisaab se 'aage badhna' kya hai?",
    "Aisi job me jana jisse aap nafrat karte ho, mentally bohot drain karta hai. Aapko kis tarah ke kaam me thodi si shanti milti hai?",
    "Apne hi faisle par doubt karna bohot bechaini deta hai. Agar koi risk na ho, to aap kya karna chahenge?"
]
CAR_RESP_GU = [
    "Bija loko ne aagal vadhta joi ne hatash thavu swabhavik che. Pan tamari potani safalta ni vyakhya shu che?",
    "Na gamti job karvi ae mansik rite thakvadi de che. Tamne kaya kaam ma anand aave che?",
    "Sanka aapna aatm-vishwas ne todi nakhe che. Jo koi dar na hoy, to tame kya marg par chalva mango cho?"
]

# 7. MOTIVATION
MOT_USER_EN = [
    "I can't find the energy to do even the basic daily tasks anymore.",
    "I have so many goals but zero drive to actually start working on them.",
    "Everything just feels pointless. Why should I even try?"
]
MOT_USER_HI = [
    "Mujhme ab basic kaam karne ki bhi energy nahi bachi hai.",
    "Mere paas goals bohot hai par unpar kaam shuru karne ki himmat nahi ho rahi.",
    "Har cheez bekar lagti hai, lagta hai koshish hi kyun karu?"
]
MOT_USER_GU = [
    "Mara ma roj na nana kaam karvani pan shakti nathi rahi.",
    "Mara ghana lakshya che pan teni par kaam sharu karvani iccha nathi thhthi.",
    "Badhu j vyarth laage che. Hu koshish j kem karu?"
]
MOT_RESP_EN = [
    "It sounds like you're carrying a very heavy exhaustion. It's okay that basic tasks are hard right now. What is the absolute bare minimum you can forgive yourself for today?",
    "Having goals but lacking the drive can make you feel guilty. I hear that struggle. What makes the very first step feel so impossible?",
    "When everything feels pointless, trying feels like moving mountains. I hear your frustration. What is one small thing that matters to you right now?"
]
MOT_RESP_HI = [
    "Aapki baaton se lag raha hai ki aap mentally bohot thak chuke hai. Abhi basic kaam na kar pana completely normal hai. Kya hum aaj ke liye sirf ek bohot chhota target rakh sakte hai?",
    "Goals hona aur drive na hona bohot guilt deta hai. Mai samajh raha hu. Wo pehla kadam lene me sabse badi rukaawat kya lag rahi hai?",
    "Jab sab kuch bekar lagne lage, toh koshish karna bhi ek pahaad uthane jaisa lagta hai. Aapko iss waqt sabse zyada kya zaroori lag raha hai?"
]
MOT_RESP_GU = [
    "Tamari vaat par thi laage che ke tame mansik rite thaki gaya cho. Nana kaam na thai shake to pan chinta na karo. Aaje tamara mate sauthi saral kaam kayu che?",
    "Lakshya hova pan iccha na thavi ae aatm-glani (guilt) aape che. Te pehlu kadam uthavva maa kya bhay laage che?",
    "Jyaare badhu j vyarth laage, tyare ek nani koshish pan pahaad jevi laage che. Tame atyare kem haraeli mehsus karo cho?"
]

# 8. SELF-ESTEEM
SLF_USER_EN = [
    "I feel so ugly and worthless compared to everyone else.",
    "No matter what I achieve, I always feel like a complete fraud.",
    "I hate myself so much. I can't look in the mirror without crying."
]
SLF_USER_HI = [
    "Mujhe apne aap se nafrat hai, dusro ke saamne mai bilkul bekar lagta hu.",
    "Mai kitna bhi achha kar lu, andar se lagta hai ki mai ek fraud hu.",
    "Mai khud se itni nafrat karta hu ki aaine me khud ko dekh kar rona aata hai."
]
SLF_USER_GU = [
    "Mane mara potana thi nafrat che, bija ni sarkhamani maa hu ekdam kharab chu.",
    "Hu game tetlu saaru karu, pan andar thi mane aevu j laage ke hu kabil nathi.",
    "Mane potana chehra thi j nafrat che, ariyasa ma joi ne mane radvu aave che."
]
SLF_RESP_EN = [
    "Hearing you call yourself worthless breaks my heart. It sounds like you are carrying a lot of self-criticism. Where do you think these harsh words originally came from?",
    "That feeling of being a fraud, despite your achievements, is incredibly heavy. What evidence do you have that proves your achievements are genuinely yours?",
    "I hear how deep that hatred runs right now, and I'm so sorry it hurts to look in the mirror. I am here with you. What part of you feels the most rejected?"
]
SLF_RESP_HI = [
    "Jab aap khud ko bekar kehte hai toh sunkar bohot dukh hota hai. Ye khud ki aalochana karne ki aadat kahan se shuru hui thi?",
    "Achha karne ke baad bhi khud par shaq karna bohot bhari padta hai. Aapki iss success me aapki khud ki mehnat kitni thi?",
    "Aapki khud se narazgi bohot gehri hai, aur aaine me na dekh paana sach me dardnaak hai. Main yahan hu. Aap apne andar kis cheez ko accept nahi kar pa rahe?"
]
SLF_RESP_GU = [
    "Jyaare tame potani jaat ne kharab kaho cho tyare bau dukh thay che. Aatli badhi self-criticism kya thi aavi?",
    "Safalta malya pachi pan potana par shanka karvi bau dukhdayak che. Tamari safalta pachal tamari mehnat ketli hati?",
    "Potana thi nafrat karvi ane radvu aavvu ae ghanu pidayadayak che. Hu tamari saathe chu. Tame potani kai vaat thi sauthi vadhu naraj cho?"
]


# Combine them into a structured dict
DATA_POOLS = {
    "Exam stress": (EXAM_USER_EN + EXAM_USER_HI + EXAM_USER_GU, EXAM_RESP_EN + EXAM_RESP_HI + EXAM_RESP_GU),
    "Breakup": (BRK_USER_EN + BRK_USER_HI + BRK_USER_GU, BRK_RESP_EN + BRK_RESP_HI + BRK_RESP_GU),
    "Anxiety": (ANX_USER_EN + ANX_USER_HI + ANX_USER_GU, ANX_RESP_EN + ANX_RESP_HI + ANX_RESP_GU),
    "Loneliness": (LON_USER_EN + LON_USER_HI + LON_USER_GU, LON_RESP_EN + LON_RESP_HI + LON_RESP_GU),
    "Family pressure": (FAM_USER_EN + FAM_USER_HI + FAM_USER_GU, FAM_RESP_EN + FAM_RESP_HI + FAM_RESP_GU),
    "Career confusion": (CAR_USER_EN + CAR_USER_HI + CAR_USER_GU, CAR_RESP_EN + CAR_RESP_HI + CAR_RESP_GU),
    "Motivation": (MOT_USER_EN + MOT_USER_HI + MOT_USER_GU, MOT_RESP_EN + MOT_RESP_HI + MOT_RESP_GU),
    "Self-esteem": (SLF_USER_EN + SLF_USER_HI + SLF_USER_GU, SLF_RESP_EN + SLF_RESP_HI + SLF_RESP_GU)
}

# Add modifiers to exponentially increase combinations (adding emotional intensity context)
PREFIXES = [
    "", "Honestly, ", "I don't know who to tell but, ", 
    "Please help me, ", "I am losing my mind. ", "Lately, ",
    "To be honest, ", "I've been feeling this for a while: ",
    "Listen, ", "I just need to vent. ", "I feel so lost. ",
    "Can someone tell me what to do? ", "It's getting worse... "
]

SUFFIXES = [
    "", " It's just so hard.", " I don't know what to do.",
    " Does this make sense?", " Am I crazy?", " I'm so tired of this.",
    " Please just listen.", " What is wrong with me?", " I can't take it anymore.",
    " Has anyone else felt this way?", " It hurts so much."
]

def generate_pairs():
    print("Generating 40,000 deep psychological pairs...")
    
    # Load existing
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    else:
        dataset = []
    
    print(f"Current dataset size: {len(dataset)}")
    
    generated = []
    
    for category, (users, resps) in DATA_POOLS.items():
        pairs_for_cat = 0
        
        # Combinatorial explosion
        for u in users:
            for r in resps:
                for pre in PREFIXES:
                    for suf in SUFFIXES:
                        if pairs_for_cat >= 5000:
                            break
                        
                        input_text = pre + u.lower() if pre else u
                        input_text += suf
                        
                        generated.append({
                            "input": input_text,
                            "response": r
                        })
                        pairs_for_cat += 1
                if pairs_for_cat >= 5000:
                    break
    
    print(f"Generated {len(generated)} new pairs.")
    
    # Combine and deduplicate
    all_data = dataset + generated
    
    # Simple Deduplication
    seen = set()
    unique_data = []
    for d in all_data:
        # deduplicate based on the exact pair
        pair_key = (d["input"].strip().lower(), d["response"].strip().lower())
        if pair_key not in seen:
            seen.add(pair_key)
            unique_data.append(d)
            
    print(f"Final dataset size after deduplication: {len(unique_data)}")
    
    random.shuffle(unique_data)
    
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(unique_data, f, indent=2, ensure_ascii=False)
        
    print("Dataset saved successfully!")

if __name__ == "__main__":
    generate_pairs()
