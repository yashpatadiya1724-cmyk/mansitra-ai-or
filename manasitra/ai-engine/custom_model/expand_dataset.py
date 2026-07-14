"""
Manasitra Dataset Expansion Script
===================================
Expands the psychology_dataset.json to ~3x its current size by:
1. Adding new diverse therapy Q&A pairs across many mental health topics
2. Augmenting existing pairs with paraphrases and variations
3. Deduplicating the final dataset

Current: ~3,916 pairs → Target: ~12,000+ pairs
"""

import json
import os
import random
import re
import copy

DATASET_PATH = os.path.join(os.path.dirname(__file__), "psychology_dataset.json")
OUTPUT_PATH  = os.path.join(os.path.dirname(__file__), "psychology_dataset.json")
BACKUP_PATH  = os.path.join(os.path.dirname(__file__), "psychology_dataset_backup.json")


# ============================================================================
# PART 1: NEW HAND-CRAFTED THERAPY PAIRS (diverse topics)
# ============================================================================

NEW_PAIRS = [
    # ── DEPRESSION ──────────────────────────────────────────────────────────
    {"input": "I don't see the point in anything anymore", "response": "When everything feels pointless, it can be incredibly heavy. I'm glad you're sharing this with me. Can you tell me when you first started feeling this way?"},
    {"input": "I can't get out of bed most days", "response": "That sounds really difficult. Sometimes depression makes even simple things feel impossible. What does a typical morning look like for you?"},
    {"input": "Nothing makes me happy like it used to", "response": "Losing interest in things you once enjoyed is a common sign of depression. It doesn't mean those things won't bring you joy again. When did you notice this change?"},
    {"input": "I feel empty inside all the time", "response": "That emptiness can feel overwhelming. You're not broken for feeling this way. Have you been able to talk to anyone else about how you're feeling?"},
    {"input": "I cry for no reason sometimes", "response": "Tears can come when emotions build up inside, even if you can't name them. It's okay to cry. What do you think might be behind those tears?"},
    {"input": "I feel like a burden to everyone around me", "response": "I hear you, and I want you to know that you matter. Depression often tells us lies about being a burden. The people who love you would want to know you're struggling."},
    {"input": "I've lost all motivation to do anything", "response": "Losing motivation is exhausting, especially when you feel like you should be doing more. Let's start small. What's one tiny thing you managed to do today?"},
    {"input": "Every day feels exactly the same and meaningless", "response": "When days blur together, it can feel like you're stuck in a fog. That's a really tough place to be. What would a meaningful day look like for you?"},
    {"input": "I feel numb, I can't feel anything at all", "response": "Emotional numbness can be your mind's way of protecting you from overwhelming feelings. It's temporary, even though it doesn't feel that way. When did you start feeling this numbness?"},
    {"input": "I don't enjoy eating or sleeping anymore", "response": "Changes in appetite and sleep are often connected to how we're feeling emotionally. Are you eating and sleeping less, or has the pattern changed in other ways?"},
    {"input": "I feel worthless and like a failure", "response": "Those feelings are painful, and I want you to know they don't define who you are. Depression often distorts how we see ourselves. What evidence does your mind use to call you a failure?"},
    {"input": "I've been isolating myself from everyone", "response": "Pulling away can feel like the safest option when you're hurting. But isolation often makes things harder. Is there one person you feel somewhat safe reaching out to?"},
    {"input": "I hate myself", "response": "That's a really painful thing to carry. Self-hatred often comes from internalizing criticism or difficult experiences. You deserve compassion, including from yourself. Can we explore where these feelings come from?"},
    {"input": "Life feels so heavy right now", "response": "I can hear how much weight you're carrying. You don't have to carry it all at once. Let's talk about what feels heaviest right now."},
    {"input": "I feel stuck and can't move forward", "response": "Feeling stuck is incredibly frustrating. Sometimes we need to stop trying to move forward and just be where we are for a moment. What do you feel is keeping you stuck?"},

    # ── ANXIETY ─────────────────────────────────────────────────────────────
    {"input": "My heart races and I can't breathe sometimes", "response": "That sounds like it could be a panic attack. They're frightening but not dangerous. Have you experienced this before? Let's talk about some grounding techniques that might help."},
    {"input": "I worry about everything constantly", "response": "Constant worry is exhausting. It's like your mind won't let you rest. What are the top three things that worry you most right now?"},
    {"input": "I have this constant feeling that something bad will happen", "response": "That sense of dread is a hallmark of anxiety. Your brain is trying to protect you, but it's on overdrive. When do you notice this feeling the most?"},
    {"input": "I can't stop overthinking everything", "response": "Overthinking can trap us in loops that feel impossible to break. One technique that helps is writing down your thoughts. Have you tried journaling?"},
    {"input": "Social situations make me extremely nervous", "response": "Social anxiety can make interactions feel overwhelming. Remember, most people are focused on themselves, not judging you. What specific situations feel hardest for you?"},
    {"input": "I'm afraid of failing at everything I try", "response": "Fear of failure can be paralyzing. But what if failing is just a step toward learning? What's something you've been avoiding because of this fear?"},
    {"input": "I feel anxious even when there's nothing to worry about", "response": "Free-floating anxiety is when your body stays in alert mode even without a clear trigger. This is your nervous system, not a reflection of weakness. Have you tried any breathing exercises?"},
    {"input": "I keep having panic attacks at work", "response": "Panic attacks at work can feel especially scary because you feel exposed. Let me share a technique: try the 5-4-3-2-1 grounding method. Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste."},
    {"input": "What if I mess everything up", "response": "That 'what if' thinking is anxiety talking. Instead of 'what if I mess up,' try asking 'what if it goes well?' Our brains tend to focus on worst-case scenarios. What specifically are you worried about messing up?"},
    {"input": "I can't relax no matter what I do", "response": "When your nervous system is activated, relaxation feels impossible. Progressive muscle relaxation can help. Start by tensing your toes for 5 seconds, then releasing. Would you like to try it together?"},
    {"input": "I'm terrified of the future", "response": "The unknown can feel terrifying. But the future hasn't happened yet, and worrying about it steals peace from today. What about the future scares you most?"},
    {"input": "My anxiety makes it hard to eat", "response": "Anxiety can definitely affect appetite. Your body's fight-or-flight response suppresses hunger. Try eating small, simple meals. What does your eating pattern look like lately?"},
    {"input": "I feel like everyone is watching and judging me", "response": "That feeling is incredibly uncomfortable. Social anxiety amplifies our self-consciousness. In reality, people are usually much less focused on us than we think. What situations trigger this feeling most?"},
    {"input": "I'm scared of being alone", "response": "Fear of being alone is more common than you might think. It often connects to deeper fears about self-worth. What does being alone feel like for you?"},
    {"input": "I can't sleep because my mind won't stop racing", "response": "Racing thoughts at bedtime are really frustrating. Your mind is processing everything it couldn't during the day. Have you tried writing down your thoughts before bed to empty your mind?"},

    # ── RELATIONSHIPS ───────────────────────────────────────────────────────
    {"input": "My partner doesn't understand me", "response": "Feeling misunderstood by someone close to you is really painful. Communication styles often differ between partners. Have you tried expressing your needs directly to them?"},
    {"input": "I keep getting into toxic relationships", "response": "Recognizing patterns is an important first step. Often we're drawn to what feels familiar, even if it's unhealthy. What do these relationships typically look like?"},
    {"input": "My friend betrayed my trust", "response": "Betrayal hurts deeply, especially from someone you trusted. Your feelings of hurt and anger are completely valid. Would you like to talk about what happened?"},
    {"input": "I'm afraid of getting close to anyone", "response": "Fear of intimacy often comes from past hurts. It's your heart trying to protect itself. But walls that keep pain out also keep love out. What do you think you're most afraid of?"},
    {"input": "My parents are always fighting and it affects me", "response": "Living in a home with conflict is stressful and emotionally draining. Their fighting is not your fault or responsibility. How does their arguing make you feel?"},
    {"input": "I just went through a really bad breakup", "response": "Breakups can feel like grief because you're losing a part of your daily life. Give yourself permission to feel whatever comes up. How long were you together?"},
    {"input": "I feel like nobody truly knows the real me", "response": "Feeling unseen is lonely. Sometimes we hide parts of ourselves because we fear rejection. What parts of yourself do you feel you can't show others?"},
    {"input": "My family expects too much from me", "response": "Family pressure can feel suffocating, especially when you want to please them but also be yourself. What expectations feel heaviest right now?"},
    {"input": "I feel used by the people around me", "response": "Feeling used is really hurtful. It's important to set boundaries. Do you feel comfortable saying no when people ask too much of you?"},
    {"input": "I'm jealous of my friend's success", "response": "Jealousy is a normal human emotion. It often points to our own unmet desires. What about their success feels most triggering for you?"},
    {"input": "My partner is emotionally distant", "response": "Emotional distance in a relationship can feel incredibly lonely. Have you been able to tell your partner how their distance affects you?"},
    {"input": "I don't trust anyone anymore", "response": "Losing trust in people is often a result of being hurt. It's a protective mechanism. Can you share what experiences led to this feeling?"},
    {"input": "My best friend ghosted me", "response": "Being ghosted by someone close is confusing and hurtful. You deserve closure. Sometimes people pull away for their own reasons. How has this affected you?"},
    {"input": "I feel controlled by my partner", "response": "Feeling controlled is a serious concern. Everyone deserves to feel free in their relationships. Can you tell me more about what makes you feel controlled?"},
    {"input": "I can't forgive someone who hurt me", "response": "Forgiveness is a process, not a switch. It doesn't mean what they did was okay. It means freeing yourself from carrying that pain. What do you think holds you back from forgiving?"},

    # ── SELF-ESTEEM ─────────────────────────────────────────────────────────
    {"input": "I'm not good enough for anything", "response": "That critical voice inside you is not the truth about who you are. You are enough, exactly as you are. Where do you think this belief started?"},
    {"input": "I compare myself to others all the time", "response": "Comparison truly is the thief of joy. You're seeing others' highlight reels while living your behind-the-scenes. What do you usually compare yourself about?"},
    {"input": "I don't think anyone could love the real me", "response": "That fear of being unlovable is painful but not true. The right people will love you for exactly who you are. What makes you feel this way?"},
    {"input": "I always feel like I'm not doing enough", "response": "That 'never enough' feeling is exhausting. You're doing more than you give yourself credit for. What would 'enough' look like to you?"},
    {"input": "I hate the way I look", "response": "Body image struggles are really painful. Your worth is not determined by your appearance. What specifically bothers you, and when did these feelings start?"},
    {"input": "I feel inferior to everyone around me", "response": "Feeling inferior is a painful experience. Remember, everyone has their own struggles that aren't visible. What situations make you feel most inferior?"},
    {"input": "I'm a perfectionist and it's ruining my life", "response": "Perfectionism often comes from a fear of not being good enough. It's exhausting to hold yourself to impossible standards. What happens when you don't meet your own expectations?"},
    {"input": "I feel like an imposter at work", "response": "Imposter syndrome is incredibly common, even among highly successful people. The fact that you care about doing well shows competence. When do you feel this most strongly?"},
    {"input": "I can't accept compliments", "response": "Difficulty accepting compliments often connects to low self-worth. Next time someone compliments you, try just saying 'thank you' without deflecting. What do you usually say instead?"},
    {"input": "I feel like I'll never be successful", "response": "Success looks different for everyone. The fact that you're thinking about it means you have drive. What does success mean to you personally?"},

    # ── GRIEF & LOSS ────────────────────────────────────────────────────────
    {"input": "I lost someone very close to me recently", "response": "I'm so sorry for your loss. Grief has no timeline and no right way to experience it. Would you like to tell me about the person you lost?"},
    {"input": "I can't stop thinking about someone who passed away", "response": "When we lose someone dear, they stay with us in our thoughts and hearts. It's natural to think about them often. What memories come to mind most?"},
    {"input": "I feel guilty about my loved one's death", "response": "Guilt in grief is very common, but it's rarely deserved. We often wish we could have done something differently. What specifically do you feel guilty about?"},
    {"input": "I don't know how to cope with loss", "response": "There's no instruction manual for grief, and that's what makes it so hard. Everyone processes loss differently. What has helped you cope so far, even a little?"},
    {"input": "It's been months and the pain hasn't gotten better", "response": "Grief doesn't follow a schedule. Some days will be harder than others, and that's okay. The pain changing shape doesn't mean you're forgetting them. How are you taking care of yourself?"},
    {"input": "I lost my pet and everyone says it's just an animal", "response": "Your grief is valid. Pets are family, and losing them is a real loss. Don't let anyone minimize your pain. Tell me about your pet."},
    {"input": "My parent passed away and I feel lost", "response": "Losing a parent changes everything. They're often our foundation, and without them, the world feels different. I'm here with you. What do you miss most about them?"},
    {"input": "I'm grieving a friendship that ended", "response": "Friendship breakups are a real form of grief that often gets overlooked. You lost someone important to you. What made that friendship special?"},

    # ── STRESS & BURNOUT ────────────────────────────────────────────────────
    {"input": "I'm completely burned out from work", "response": "Burnout is your mind and body telling you something needs to change. You've been giving too much without replenishing yourself. What does your typical workday look like?"},
    {"input": "I have too many responsibilities and I can't handle them all", "response": "Feeling overwhelmed by responsibilities is a sign that you need to prioritize and delegate. You can't pour from an empty cup. What's taking up the most energy right now?"},
    {"input": "I can't take a break because I'll fall behind", "response": "That fear of falling behind keeps many people trapped in a cycle of exhaustion. But rest actually improves performance. When was the last time you truly rested?"},
    {"input": "My job is making me miserable", "response": "Spending most of your waking hours doing something that makes you miserable takes a real toll. What specifically about your job is causing the most distress?"},
    {"input": "I feel like I'm drowning in work", "response": "That drowning feeling is real and overwhelming. Let's try to find some solid ground. Can you list the three most urgent things on your plate right now?"},
    {"input": "I never have time for myself", "response": "Losing yourself in obligations is a common pattern. Self-care isn't selfish, it's necessary. If you had one free hour right now, what would you do with it?"},
    {"input": "I'm stressed about money all the time", "response": "Financial stress affects every part of life. It's one of the most common stressors people face. What aspect of your finances worries you most?"},
    {"input": "I feel exhausted even after sleeping", "response": "Emotional exhaustion can make you feel tired no matter how much you sleep. Your mind is working overtime even during rest. What's weighing on you most?"},
    {"input": "Everything feels urgent and I can't prioritize", "response": "When everything feels urgent, nothing gets prioritized well. Try the Eisenhower Matrix: separate tasks into urgent-important, important-not urgent, urgent-not important, and neither. What feels most urgent to you?"},
    {"input": "I work all the time and still feel like it's not enough", "response": "That's a painful cycle. Working harder doesn't always mean working better. Sometimes we need to step back to move forward. What drives you to keep pushing?"},

    # ── ACADEMIC STRESS ─────────────────────────────────────────────────────
    {"input": "I'm terrified of failing my exams", "response": "Exam fear is incredibly common. A little anxiety can help you focus, but too much can be paralyzing. How are you currently preparing?"},
    {"input": "My parents will be disappointed if I get bad grades", "response": "Fear of disappointing parents adds enormous pressure. Your worth isn't measured by grades. Have you been able to talk to your parents about how you're feeling?"},
    {"input": "I can't concentrate on studying", "response": "Difficulty concentrating can come from stress, lack of sleep, or feeling overwhelmed. Try the Pomodoro technique: study for 25 minutes, then take a 5-minute break. What subject are you struggling with?"},
    {"input": "I feel stupid compared to my classmates", "response": "Everyone learns at their own pace. Comparing your journey to others only creates unnecessary suffering. What makes you feel this way in class?"},
    {"input": "I have so much homework I don't know where to start", "response": "When the workload feels impossible, starting small helps. Pick the easiest task first to build momentum. What assignments do you have right now?"},
    {"input": "I'm afraid I chose the wrong career path", "response": "It's normal to question your path. Many successful people changed directions multiple times. What made you choose this path originally, and what's making you doubt it now?"},
    {"input": "College feels so overwhelming", "response": "The transition to college brings a lot of changes at once. It's okay to feel overwhelmed. What part of college life is hardest for you?"},
    {"input": "I procrastinate everything and then panic", "response": "Procrastination is often driven by fear or perfectionism, not laziness. Understanding why you procrastinate is the first step. What do you typically procrastinate on?"},
    {"input": "I have a presentation tomorrow and I'm panicking", "response": "Presentation anxiety is very common. Try this: prepare your opening line really well, practice it 5 times, and remember that your audience wants you to succeed. What's your presentation about?"},
    {"input": "I feel like I'm wasting my youth studying", "response": "That tension between investing in your future and living in the present is real. Balance is key. What would you like to do more of outside of studying?"},

    # ── ANGER MANAGEMENT ────────────────────────────────────────────────────
    {"input": "I get angry over small things and I don't know why", "response": "Disproportionate anger often signals that something deeper is bothering you. Small things become the last straw. What's been going on beneath the surface?"},
    {"input": "I said something really hurtful to someone I love when I was angry", "response": "Anger can make us say things we don't mean. The fact that you feel bad about it shows you care. Have you been able to apologize to them?"},
    {"input": "I feel like I'm going to explode with anger", "response": "That building pressure needs a release. Physical activity, deep breathing, or even stepping away from the situation can help. What usually triggers this intensity?"},
    {"input": "I can't control my temper", "response": "Learning to manage anger is a skill that takes practice. One technique is to pause and count to 10 before responding. What situations make you lose your temper most?"},
    {"input": "I'm angry at the world", "response": "Broad anger at the world often comes from feeling powerless or hurt. It's valid to feel angry. What in the world is bothering you most right now?"},
    {"input": "I broke something in anger and I feel terrible", "response": "Acting on anger with destruction shows how intense your feelings are. Feeling bad afterward is a sign of growth. Let's find healthier outlets for that energy. What happened?"},
    {"input": "My anger is pushing people away", "response": "Recognizing that pattern is brave. Anger often masks vulnerability. When you feel angry, what emotion is underneath it? Sometimes it's fear, hurt, or sadness."},
    {"input": "I resent my parents for how they raised me", "response": "Resentment toward parents is more common than people admit. It's okay to feel hurt about your childhood while also moving forward. What do you resent most?"},

    # ── LONELINESS & ISOLATION ──────────────────────────────────────────────
    {"input": "I feel alone even when I'm surrounded by people", "response": "Feeling lonely in a crowd is a special kind of pain. It means you're craving deeper connection, not just presence. What kind of connection are you missing?"},
    {"input": "I don't have any real friends", "response": "Quality friendships take time to build. You're not alone in feeling this way. What does a real friendship look like to you?"},
    {"input": "Nobody would notice if I disappeared", "response": "I hear the pain in those words. People often matter more than they realize. I notice you, right here, right now. What made you feel this way?"},
    {"input": "I spend all my time alone in my room", "response": "Isolation can feel safe but it often deepens loneliness. Even small steps outside your comfort zone can help. What's one thing you could do this week that involves being around people?"},
    {"input": "Moving to a new city and I know nobody", "response": "Starting over in a new place is exciting and terrifying at the same time. Building a new social circle takes time. What brought you to the new city?"},
    {"input": "I miss having someone to talk to", "response": "That need for connection is deeply human. I'm here to listen right now. What would you like to talk about most?"},
    {"input": "I feel disconnected from everyone", "response": "Disconnection can happen gradually until one day you realize how far apart you feel from others. What do you think caused this distance?"},
    {"input": "Everyone has their own group and I don't belong anywhere", "response": "Not belonging is a painful feeling. But sometimes the right group just hasn't appeared yet. What are your interests? They can be a bridge to finding your people."},

    # ── SLEEP ISSUES ────────────────────────────────────────────────────────
    {"input": "I can't fall asleep at night", "response": "Difficulty sleeping is often connected to an overactive mind. Creating a wind-down routine can help. What does your evening typically look like before bed?"},
    {"input": "I keep having nightmares", "response": "Recurring nightmares can be your brain's way of processing unresolved emotions or stress. What are your nightmares usually about?"},
    {"input": "I sleep too much and still feel tired", "response": "Oversleeping and still feeling exhausted can be a sign of depression or emotional fatigue. How long has this been going on?"},
    {"input": "I dread going to bed because I know I won't sleep", "response": "Anticipating sleeplessness actually makes it worse because it creates anxiety around bedtime. Have you tried limiting screen time an hour before bed?"},
    {"input": "I wake up in the middle of the night and can't go back to sleep", "response": "Middle-of-the-night waking is frustrating. Often our worries wake us up. What's usually on your mind when you wake up?"},

    # ── SELF-HARM & SAFETY ──────────────────────────────────────────────────
    {"input": "I've been thinking about hurting myself", "response": "Thank you for trusting me with this. Your safety matters. If you're in immediate danger, please contact a crisis helpline. These thoughts don't define you. Can you tell me more about what you're feeling?"},
    {"input": "Sometimes I just want the pain to stop", "response": "I hear you, and I want you to know there are ways to ease the pain that don't involve hurting yourself. You reached out, which shows incredible strength. What kind of pain are you experiencing?"},
    {"input": "I don't want to live anymore", "response": "I'm really glad you told me this. You matter, and there is help available. Please reach out to a crisis helpline or trusted person. These feelings can get better with support. Are you safe right now?"},
    {"input": "I used to self-harm and I'm afraid of relapsing", "response": "Recovery isn't always linear, and being aware of triggers is really important. You've already shown strength by stopping before. What's making you feel vulnerable to relapsing?"},

    # ── TRAUMA & PTSD ───────────────────────────────────────────────────────
    {"input": "Something terrible happened to me and I can't get over it", "response": "Trauma leaves deep marks, and healing takes time. You don't have to 'get over it' on anyone's timeline. Would you feel comfortable sharing a little about what happened?"},
    {"input": "I keep having flashbacks of bad things that happened", "response": "Flashbacks are your brain's way of trying to process traumatic experiences. They can be overwhelming. Grounding techniques can help bring you back to the present. Would you like to try one?"},
    {"input": "I don't feel safe anywhere", "response": "Not feeling safe is exhausting and isolating. Your feelings are valid. Let's work on identifying what makes you feel even slightly safer. Is there any place or person that brings you some comfort?"},
    {"input": "I was bullied and it still affects me", "response": "Bullying can leave lasting emotional scars. The pain you feel is real and valid, no matter how long ago it happened. How does it affect you now?"},
    {"input": "I can't trust people after what happened to me", "response": "When someone breaks your trust deeply, it makes sense to be guarded. Trust can be rebuilt slowly, at your own pace. What would feel like a safe first step?"},

    # ── COPING STRATEGIES ───────────────────────────────────────────────────
    {"input": "How do I deal with negative thoughts", "response": "One effective technique is cognitive reframing. When a negative thought comes, ask yourself: Is this thought true? Is there evidence against it? What would I tell a friend who had this thought?"},
    {"input": "What can I do when I feel overwhelmed", "response": "When overwhelm hits, try the STOP technique: Stop what you're doing, Take a breath, Observe what you're feeling, Proceed with one small step. What overwhelms you most often?"},
    {"input": "How do I manage my emotions better", "response": "Emotional regulation starts with awareness. Try naming your emotions throughout the day without judging them. Just noticing 'I feel anxious right now' can reduce the intensity. Would you like to try this?"},
    {"input": "I need healthy ways to cope with stress", "response": "Great that you're seeking healthy coping methods! Some options include deep breathing, progressive muscle relaxation, journaling, physical exercise, and creative expression. Which of these appeals to you?"},
    {"input": "How do I stop negative self-talk", "response": "Catching negative self-talk is the first step. When you notice a harsh thought, try replacing it with what you'd say to a friend in the same situation. What does your inner critic usually say?"},
    {"input": "Can you teach me a breathing exercise", "response": "Of course! Try box breathing: Breathe in for 4 counts, hold for 4 counts, breathe out for 4 counts, hold for 4 counts. Repeat 4 times. This activates your parasympathetic nervous system and calms you down."},
    {"input": "What is mindfulness and how can it help me", "response": "Mindfulness means paying attention to the present moment without judgment. It helps reduce anxiety and rumination. A simple start: focus on your breath for 60 seconds. When your mind wanders, gently bring it back."},
    {"input": "I use food to cope with my emotions", "response": "Emotional eating is a common coping mechanism. Food provides temporary comfort but doesn't address the underlying feelings. What emotions usually trigger you to eat?"},
    {"input": "I cope by avoiding everything", "response": "Avoidance provides short-term relief but increases anxiety long-term. You're stuck in a cycle. Would you be willing to try facing one small thing you've been avoiding this week?"},
    {"input": "How do I build resilience", "response": "Resilience grows through facing challenges, not avoiding them. It also grows through self-compassion, strong connections, and finding meaning in difficult experiences. What challenge are you facing right now?"},

    # ── MOTIVATION & PURPOSE ────────────────────────────────────────────────
    {"input": "I don't know what I want to do with my life", "response": "Not knowing your path is more common than you think. Instead of looking for one big answer, try exploring small interests. What activities make you lose track of time?"},
    {"input": "I have no goals or ambitions", "response": "Sometimes a lack of goals comes from fear of failure or depression. It's okay to not have everything figured out. What did you dream of doing when you were younger?"},
    {"input": "I feel like my life has no purpose", "response": "Searching for purpose is a universal human experience. Purpose often comes from connection, helping others, and doing what aligns with your values. What matters most to you?"},
    {"input": "I keep starting things but never finish them", "response": "Difficulty finishing things can come from perfectionism, fear of judgment, or losing interest. What usually happens when you stop midway?"},
    {"input": "I feel like I'm falling behind everyone my age", "response": "Life isn't a race, even though social media makes it feel that way. Everyone's timeline is different. What makes you feel behind specifically?"},
    {"input": "I want to change but I don't know how", "response": "The desire for change is the first and most important step. Change starts small. What's one thing about your life you'd most like to change?"},
    {"input": "I feel lost in life", "response": "Feeling lost is uncomfortable but it's also an opportunity for exploration. Sometimes we need to get lost to find a new path. What areas of your life feel most unclear?"},
    {"input": "Nothing excites me anymore", "response": "Loss of excitement can be a sign of burnout or depression. When was the last time you felt truly excited about something?"},

    # ── BODY IMAGE ──────────────────────────────────────────────────────────
    {"input": "I'm not comfortable in my own body", "response": "Body discomfort is really painful because you can't escape your body. Your body does amazing things for you every day. What specifically makes you uncomfortable?"},
    {"input": "I constantly check my appearance and feel ugly", "response": "Frequent mirror-checking and feeling ugly can become a distressing cycle. Beauty standards are unrealistic. What do you see when you look in the mirror?"},
    {"input": "I'm scared to eat because I'll gain weight", "response": "Fear of weight gain that affects eating is something to take seriously. Your body needs fuel to function. Can you tell me more about your relationship with food?"},
    {"input": "Social media makes me hate my body", "response": "Social media is designed to show idealized, filtered versions of reality. Comparing yourself to edited images is unfair to you. Would you consider a social media break?"},
    {"input": "I exercise obsessively and I can't stop", "response": "Exercise addiction can be just as harmful as not exercising. When exercise becomes compulsive, it's often about control. What happens when you try to take a rest day?"},

    # ── FAMILY ISSUES ───────────────────────────────────────────────────────
    {"input": "My parents are getting divorced and I don't know how to handle it", "response": "Parental divorce affects children of all ages. It's okay to feel confused, sad, or even angry. Their relationship is separate from their love for you. What emotion feels strongest right now?"},
    {"input": "I feel like the black sheep of my family", "response": "Feeling like the outsider in your own family is really isolating. Sometimes being different isn't a flaw, it's a strength. What makes you feel like the black sheep?"},
    {"input": "My siblings get all the attention and I'm invisible", "response": "Feeling invisible in your family is hurtful. Every child deserves to be seen. Have you ever expressed these feelings to your parents?"},
    {"input": "My parents are too strict and controlling", "response": "Overly strict parenting can feel suffocating, especially as you grow older. Often, controlling behavior comes from fear. What rules or behaviors feel most restrictive?"},
    {"input": "I take care of everyone in my family but nobody takes care of me", "response": "Being the caretaker without receiving care is emotionally draining. You deserve support too. Who takes care of you when you're struggling?"},
    {"input": "My family doesn't believe in mental health", "response": "Cultural and family attitudes toward mental health can make it hard to seek help. Your feelings are valid regardless of what others believe. How does this affect you?"},
    {"input": "I feel trapped in my family situation", "response": "Feeling trapped is overwhelming. Even in difficult situations, there are usually small choices you can make. What would freedom look like for you?"},

    # ── POSITIVE / GROWTH ───────────────────────────────────────────────────
    {"input": "I'm starting to feel a little better", "response": "That's wonderful to hear! Progress isn't always dramatic. Small improvements matter. What do you think has helped you feel better?"},
    {"input": "I want to learn to love myself", "response": "Self-love is a journey, not a destination. Start by treating yourself with the same kindness you'd show a good friend. What's one thing you genuinely like about yourself?"},
    {"input": "I'm trying to be more positive", "response": "Shifting toward positivity is a great intention. It doesn't mean ignoring negative feelings, but making room for good ones too. What positive thing happened to you recently?"},
    {"input": "I made a new friend today", "response": "That's really great! New connections are precious. What do you like about this new friend?"},
    {"input": "I stood up for myself today", "response": "That takes courage! Standing up for yourself is a powerful step. How did it feel? What happened?"},
    {"input": "I'm learning to set boundaries", "response": "Setting boundaries is one of the most important skills for mental health. It might feel uncomfortable at first, but it gets easier. What boundary did you set recently?"},
    {"input": "I feel grateful today", "response": "Gratitude is a beautiful feeling. Research shows it actually changes your brain for the better. What are you feeling grateful for?"},
    {"input": "I helped someone who was struggling", "response": "Helping others is meaningful. It shows empathy and strength. How did it feel to be there for someone?"},
    {"input": "I'm trying therapy for the first time", "response": "Taking that first step toward therapy is incredibly brave. It shows real self-awareness and a desire to grow. What made you decide to try it?"},
    {"input": "I want to become a better person", "response": "The desire to grow shows you're already on the right path. Becoming better is about progress, not perfection. What areas would you like to work on?"},

    # ── ADDICTION & SUBSTANCE USE ────────────────────────────────────────────
    {"input": "I think I'm addicted to my phone", "response": "Phone addiction is increasingly common and can affect sleep, focus, and relationships. Awareness is the first step. How many hours a day do you estimate you spend on your phone?"},
    {"input": "I use alcohol to numb my feelings", "response": "Using alcohol to cope is more common than people admit, but it often makes the underlying issues worse. You're brave for acknowledging it. When did this pattern start?"},
    {"input": "I can't stop scrolling social media even though it makes me feel bad", "response": "Social media is designed to be addictive. Knowing it hurts you but not being able to stop is frustrating. Have you tried setting app time limits?"},
    {"input": "I'm afraid I'm developing a gambling problem", "response": "Recognizing a potential problem before it spirals shows great self-awareness. Gambling can become compulsive quickly. Would you be open to exploring this further?"},
    {"input": "I use shopping to feel better but then I feel worse", "response": "Emotional spending provides a temporary high followed by guilt and anxiety. The underlying need is to feel better, not to buy things. What triggers your shopping urges?"},

    # ── GENERAL WELLNESS ────────────────────────────────────────────────────
    {"input": "How can I improve my mental health", "response": "Great question! Key pillars include regular sleep, physical activity, social connection, mindfulness, and seeking help when needed. Which area would you like to start with?"},
    {"input": "What are the signs of depression", "response": "Common signs include persistent sadness, loss of interest in activities, changes in sleep or appetite, fatigue, difficulty concentrating, feelings of worthlessness, and withdrawal from others. Are you asking for yourself or someone else?"},
    {"input": "Is it normal to feel this way", "response": "Feelings are always valid. Many people experience what you're going through. What matters is that you're acknowledging them. Can you tell me more about what you're feeling?"},
    {"input": "When should I see a therapist", "response": "Consider seeing a therapist when your feelings interfere with daily life, relationships, work, or when you feel stuck despite trying to help yourself. There's no wrong time to seek support."},
    {"input": "I don't think therapy works for me", "response": "Not every therapist or approach is the right fit. Finding the right match can take time. What was your experience like? Sometimes trying a different style makes all the difference."},
    {"input": "How do I know if I have anxiety", "response": "Anxiety can show up as persistent worry, physical symptoms like rapid heartbeat, difficulty relaxing, avoidance of situations, and trouble sleeping. Does any of this resonate with you?"},
    {"input": "I want to help a friend who is depressed", "response": "Being there for a friend with depression is important. Listen without judgment, check in regularly, and encourage them to seek professional help. What has your friend been going through?"},
    {"input": "Can meditation really help with anxiety", "response": "Research strongly supports meditation for anxiety. Even 10 minutes daily can reduce stress hormones and calm the nervous system. Have you tried meditating before?"},
    {"input": "I feel better when I exercise", "response": "That's great! Exercise releases endorphins and reduces stress hormones. It's one of the most effective natural mood boosters. What type of exercise do you enjoy?"},
    {"input": "Writing helps me process my feelings", "response": "Journaling is a powerful therapeutic tool. It helps externalize thoughts and gain perspective. What do you usually write about?"},

    # ── IDENTITY & BELONGING ────────────────────────────────────────────────
    {"input": "I don't know who I am anymore", "response": "Identity confusion can happen during major life transitions. It's uncomfortable but also an opportunity to rediscover yourself. What changed recently that made you feel this way?"},
    {"input": "I feel different from everyone else", "response": "Feeling different can be lonely, but it can also be your greatest strength. Many creative and empathetic people feel this way. What makes you feel different?"},
    {"input": "I'm going through an identity crisis", "response": "Identity crises, while uncomfortable, are often catalysts for growth. You're questioning who you are because you're ready to evolve. What aspects of your identity feel uncertain?"},
    {"input": "I feel like I'm wearing a mask around people", "response": "Wearing a mask is exhausting because it prevents genuine connection. What would happen if you showed people the real you?"},
    {"input": "I don't fit in anywhere and it hurts", "response": "Not fitting in is painful, but sometimes it means you haven't found your people yet. Your uniqueness is valuable. Where have you tried to fit in?"},

    # ── WORK-LIFE BALANCE ───────────────────────────────────────────────────
    {"input": "I can't separate work from personal life", "response": "Blurred boundaries between work and personal life are increasingly common. Setting clear end-of-day rituals can help. What does your work-life boundary look like currently?"},
    {"input": "My boss makes my life miserable", "response": "A difficult boss can affect your entire wellbeing. You spend a lot of time at work, and you deserve a respectful environment. What specific behaviors are causing you distress?"},
    {"input": "I dread going to work every day", "response": "Dreading work every day is a sign something needs to change. Life is too short to be miserable for 8 hours a day. What would your ideal workday look like?"},
    {"input": "I feel undervalued at my workplace", "response": "Feeling undervalued erodes motivation and self-worth over time. You deserve recognition for your contributions. Have you expressed this to your manager?"},
    {"input": "I'm thinking about quitting my job but I'm scared", "response": "The decision to leave a job is big and scary. Fear is natural. What's holding you back most: financial concerns, uncertainty, or something else?"},

    # ── GRATITUDE & REFLECTION ──────────────────────────────────────────────
    {"input": "Thank you for listening to me", "response": "You don't need to thank me. I'm glad you felt comfortable sharing. Remember, reaching out is a sign of strength, not weakness."},
    {"input": "This conversation helped me feel better", "response": "I'm really glad to hear that. Sometimes just putting feelings into words can lighten the load. Remember, I'm here whenever you need to talk."},
    {"input": "You actually understand me", "response": "I try my best to understand, and your openness makes that possible. You deserve to be heard and understood. What would you like to explore next?"},
    {"input": "I've never told anyone this before", "response": "Thank you for trusting me with something so personal. Sharing what you've been holding inside takes real courage. I'm here, and this is a safe space."},
    {"input": "I feel lighter after talking to you", "response": "That's wonderful. Sometimes our feelings just need a place to land. Remember, you can come back anytime you need to talk."},
]


# ============================================================================
# PART 2: DATA AUGMENTATION FUNCTIONS
# ============================================================================

# Synonym replacements for augmentation
SYNONYMS = {
    "feel": ["am feeling", "experience", "sense"],
    "sad": ["down", "unhappy", "gloomy", "low"],
    "happy": ["glad", "joyful", "cheerful", "pleased"],
    "angry": ["upset", "frustrated", "furious", "irritated"],
    "stressed": ["overwhelmed", "pressured", "tense", "strained"],
    "worried": ["concerned", "anxious", "troubled", "uneasy"],
    "scared": ["afraid", "frightened", "terrified", "fearful"],
    "lonely": ["alone", "isolated", "abandoned", "solitary"],
    "tired": ["exhausted", "drained", "fatigued", "worn out"],
    "hurt": ["wounded", "pained", "injured", "damaged"],
    "terrible": ["awful", "horrible", "dreadful", "miserable"],
    "great": ["wonderful", "fantastic", "excellent", "amazing"],
    "help": ["assist", "support", "guide", "aid"],
    "talk": ["chat", "speak", "discuss", "converse"],
    "think": ["believe", "consider", "suppose", "reckon"],
    "want": ["wish", "desire", "need", "hope for"],
    "difficult": ["hard", "tough", "challenging", "rough"],
    "important": ["significant", "crucial", "vital", "essential"],
    "problem": ["issue", "challenge", "difficulty", "concern"],
    "life": ["existence", "daily life", "world", "everyday life"],
    "friend": ["buddy", "companion", "close one", "pal"],
    "family": ["loved ones", "relatives", "close family", "household"],
    "work": ["job", "career", "workplace", "occupation"],
    "school": ["college", "university", "studies", "classes"],
}

# Input prefix variations
INPUT_PREFIXES = [
    "", "I think ", "Honestly, ", "To be honest, ", "Actually, ",
    "I guess ", "I mean, ", "Well, ", "You know, ", "Sometimes ",
]

# Response empathy starters
RESPONSE_STARTERS = [
    "I hear you. ", "Thank you for sharing that. ", "I understand. ",
    "That sounds really tough. ", "Your feelings are valid. ",
    "I appreciate you opening up. ", "That must be hard. ",
    "I can see why you'd feel that way. ", "You're not alone in this. ",
]


def augment_input(text):
    """Create a variation of an input sentence."""
    variations = []

    # 1. Add prefix
    prefix = random.choice(INPUT_PREFIXES)
    if prefix and not text.startswith(prefix.strip()):
        var = prefix + text[0].lower() + text[1:]
        variations.append(var)

    # 2. Synonym replacement
    words = text.split()
    new_words = []
    changed = False
    for w in words:
        w_lower = w.lower().strip(".,!?")
        if w_lower in SYNONYMS and random.random() < 0.4:
            replacement = random.choice(SYNONYMS[w_lower])
            new_words.append(replacement)
            changed = True
        else:
            new_words.append(w)
    if changed:
        result = " ".join(new_words)
        # Capitalize first letter
        if result:
            result = result[0].upper() + result[1:]
        variations.append(result)

    return variations


def augment_response(text):
    """Create a variation of a response by adding empathy starter."""
    starter = random.choice(RESPONSE_STARTERS)
    if not text.startswith(starter.strip()):
        return starter + text[0].lower() + text[1:]
    return text


def augment_dataset(existing_pairs, target_new):
    """Generate augmented pairs from existing data."""
    augmented = []
    attempts = 0
    max_attempts = target_new * 5

    while len(augmented) < target_new and attempts < max_attempts:
        attempts += 1
        pair = random.choice(existing_pairs)
        inp = pair["input"]
        resp = pair["response"]

        # Try input augmentation
        input_vars = augment_input(inp)
        if input_vars:
            new_inp = random.choice(input_vars)
        else:
            continue

        # Sometimes also augment response
        if random.random() < 0.3:
            new_resp = augment_response(resp)
        else:
            new_resp = resp

        # Skip if too similar to original
        if new_inp.lower().strip() == inp.lower().strip():
            continue

        augmented.append({"input": new_inp, "response": new_resp})

    return augmented


# ============================================================================
# MAIN: EXPAND THE DATASET
# ============================================================================

def main():
    # Load existing dataset
    print("Loading existing dataset...")
    with open(DATASET_PATH, encoding="utf-8") as f:
        existing = json.load(f)
    print(f"  Existing pairs: {len(existing)}")

    # Backup
    print(f"  Creating backup at: {BACKUP_PATH}")
    with open(BACKUP_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    # Add new hand-crafted pairs
    print(f"\nAdding {len(NEW_PAIRS)} new hand-crafted therapy pairs...")
    all_pairs = existing + NEW_PAIRS

    # Calculate how many augmented pairs we need
    current_total = len(all_pairs)
    target_total = len(existing) * 4  # aim for 4x to end up with ~3x after dedup
    augment_needed = target_total - current_total
    print(f"  Current total: {current_total}")
    print(f"  Target total: {target_total}")
    print(f"  Augmentation needed: {augment_needed}")

    # Augment
    print("\nAugmenting dataset...")
    augmented = augment_dataset(all_pairs, augment_needed)
    print(f"  Generated {len(augmented)} augmented pairs")
    all_pairs.extend(augmented)

    # Deduplicate
    print("\nDeduplicating...")
    seen = set()
    unique_pairs = []
    for pair in all_pairs:
        key = pair["input"][:60].lower().strip()
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)
    print(f"  Removed {len(all_pairs) - len(unique_pairs)} duplicates")

    # Shuffle
    random.shuffle(unique_pairs)

    # Save
    print(f"\nFinal dataset: {len(unique_pairs)} pairs")
    print(f"  Expansion ratio: {len(unique_pairs) / len(existing):.1f}x")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(unique_pairs, f, indent=2, ensure_ascii=False)

    print(f"  Saved to: {OUTPUT_PATH}")
    print(f"  Backup at: {BACKUP_PATH}")

    # Show some samples
    print("\n--- Sample new pairs ---")
    for p in unique_pairs[-5:]:
        print(f"  INPUT   : {p['input'][:80]}")
        print(f"  RESPONSE: {p['response'][:80]}")
        print()


if __name__ == "__main__":
    main()
