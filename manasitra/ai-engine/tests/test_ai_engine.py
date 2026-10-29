"""
Comprehensive Unit Test Suite for Manasitra Emotional AI Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tests:
  1. Emotion classification (English + Hinglish, negations, safety overrides)
  2. Privacy Engine (Email, Phone, Indian PAN cards, UPI IDs, fingerprinting)
  3. Memory Engine (EAMC retention decay, state machine transitions, indexes)
  4. Conversation Brain (Hinglish crisis detection, response rules, coherence)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
import unittest
from pathlib import Path

# Add ai-engine to system path
AI_ENGINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_ENGINE_DIR))

from emotion_model.predictor import predict_emotion
from privacy_engine.privacy import privacy_filter, privacy_metadata
from memory_engine.memory import (
    _compute_retention,
    EMOTION_TRANSITIONS,
    get_or_create_user,
    save_mood,
    get_mood_history,
    record_emotion_transition,
    get_current_transition,
)
from response_engine.conversation_brain import is_crisis, _reply_breaks_emotional_rules, get_custom_model_info
from model import MODEL_PROFILES, get_model_profile, build_model


class TestEmotionPredictor(unittest.TestCase):
    def test_english_emotion_detection(self):
        cases = [
            ("I feel so joyful and proud today!", "happy"),
            ("I lost my pet and cannot stop crying", "sad"),
            ("I am furious and hate how they treated me", "angry"),
            ("Deadlines are piling up and exams are tomorrow", "stress"),
            ("I am terrified and panicking about what if scenarios", "anxiety"),
            ("Nobody talks to me, I feel completely alone", "lonely"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                res = predict_emotion(text)
                self.assertEqual(res["emotion"], expected)
                self.assertGreater(res["confidence"], 0.0)

    def test_hinglish_emotion_detection(self):
        cases = [
            ("aaj bahut udaas hu aur rona aa raha hai", "sad"),
            ("bahut tension aur pressure hai studies ka", "stress"),
            ("koi mera dost nahi hai, akela lag raha hai", "lonely"),
            ("aaj mai bahut khush hu", "happy"),
            ("mujhe bahut gussa aa raha hai", "angry"),
            ("dil me bohot ghabrahat aur bechaini hai", "anxiety"),
            ("aaj mera mood bohot off hai", "sad"),
            ("bohot bura lag raha hai", "sad"),
            ("aaj din bohot accha tha maza aaya", "happy"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                res = predict_emotion(text)
                self.assertEqual(res["emotion"], expected)

    def test_negation_handling(self):
        # "not happy" should not be classified as happy, must accurately map to sad
        res = predict_emotion("I am not happy with what happened")
        self.assertEqual(res["emotion"], "sad")

        res_hi = predict_emotion("aaj mai bilkul khush nahi hu")
        self.assertEqual(res_hi["emotion"], "sad")

        res_ok = predict_emotion("mai theek nahi hu")
        self.assertEqual(res_ok["emotion"], "sad")

    def test_distress_safety_override(self):
        # Even if a positive word appears, distress words must prevent 'happy'
        res = predict_emotion("I wish I could smile but I feel broken and hurt inside")
        self.assertNotEqual(res["emotion"], "happy")


class TestPrivacyEngine(unittest.TestCase):
    def test_email_and_phone_redaction(self):
        text = "Reach me at contact@example.com or call 9876543210"
        res = privacy_filter(text)
        self.assertIn("[REDACTED_EMAIL]", res.redacted_text)
        self.assertIn("[REDACTED_PHONE]", res.redacted_text)
        self.assertEqual(res.redaction_counts.get("email"), 1)
        self.assertEqual(res.redaction_counts.get("phone"), 1)

    def test_indian_pan_card_redaction(self):
        text = "My income tax PAN is ABCDE1234F for verification"
        res = privacy_filter(text)
        self.assertIn("[REDACTED_PAN_CARD]", res.redacted_text)
        self.assertNotIn("ABCDE1234F", res.redacted_text)

    def test_indian_upi_id_redaction(self):
        text = "Send money to student@okhdfcbank or rahul@paytm"
        res = privacy_filter(text)
        self.assertIn("[REDACTED_UPI_ID]", res.redacted_text)
        self.assertNotIn("student@okhdfcbank", res.redacted_text)
        self.assertNotIn("rahul@paytm", res.redacted_text)

    def test_privacy_metadata(self):
        text = "Personal info test"
        res = privacy_filter(text)
        meta = privacy_metadata(res)
        self.assertTrue(meta["local_only"])
        self.assertFalse(meta["external_api_used"])
        self.assertEqual(len(meta["message_fingerprint"]), 16)


class TestMemoryEngine(unittest.TestCase):
    def test_eamc_retention_decay(self):
        # Distress emotions must have higher retention than neutral after 24 hours
        sad_score = _compute_retention("sad", hours_ago=24)
        neutral_score = _compute_retention("neutral", hours_ago=24)
        self.assertGreater(sad_score, neutral_score)

    def test_emotion_transitions(self):
        self.assertEqual(EMOTION_TRANSITIONS.get(("happy", "sad")), "abrupt")
        self.assertEqual(EMOTION_TRANSITIONS.get(("sad", "neutral")), "improving")
        self.assertEqual(EMOTION_TRANSITIONS.get(("neutral", "sad")), "concerning")
        self.assertEqual(EMOTION_TRANSITIONS.get(("stress", "anxiety")), "natural")

    def test_user_creation_and_mood_save(self):
        uid = get_or_create_user("unit_test_user")
        self.assertIsInstance(uid, int)
        save_mood(uid, "happy", "Had a good day")
        history = get_mood_history(uid, limit=1)
        self.assertGreaterEqual(len(history), 1)
        self.assertEqual(history[0]["emotion"], "happy")


class TestConversationBrain(unittest.TestCase):
    def test_english_crisis_detection(self):
        self.assertTrue(is_crisis("I want to end my life"))
        self.assertTrue(is_crisis("I have no reason to live anymore"))
        self.assertFalse(is_crisis("I had a great lunch today"))

    def test_hinglish_crisis_detection(self):
        self.assertTrue(is_crisis("lagta hai zindagi khatam hai"))
        self.assertTrue(is_crisis("mai jaan de dunga"))
        self.assertTrue(is_crisis("marne ka man kar raha hai"))
        self.assertFalse(is_crisis("aaj mausam bohot accha hai"))

    def test_emotional_rule_checks(self):
        # Happy emoji should be rejected if user is in deep distress
        breaks_rule = _reply_breaks_emotional_rules(
            reply="Don't worry, cheer up! 😊",
            emotion="grief",
            user_message="My dog passed away",
        )
        self.assertTrue(breaks_rule)

    def test_custom_model_cached_metadata(self):
        info = get_custom_model_info()
        self.assertIn("ready", info)
        self.assertIn("epoch", info)
        self.assertIsInstance(info["ready"], bool)
        self.assertIsInstance(info["epoch"], int)

    def test_multilingual_crisis_keywords(self):
        # Hindi, Hinglish, and regional crisis phrases
        self.assertTrue(is_crisis("khud ko maarne ka mann kar raha hai"))
        self.assertTrue(is_crisis("jeena nahi chahta mai"))
        self.assertFalse(is_crisis("aaj assignment submit kiya"))


class TestModelArchitectureScaling(unittest.TestCase):
    def test_model_profiles_definitions(self):
        self.assertIn("small", MODEL_PROFILES)
        self.assertIn("base", MODEL_PROFILES)
        self.assertIn("large", MODEL_PROFILES)
        self.assertIn("xlarge", MODEL_PROFILES)
        self.assertEqual(MODEL_PROFILES["base"]["enc_layers"], 8)
        self.assertEqual(MODEL_PROFILES["large"]["enc_layers"], 12)
        self.assertEqual(MODEL_PROFILES["xlarge"]["enc_layers"], 16)

    def test_build_model_factory(self):
        model_s = build_model("small", vocab_size=1000, init_weights=False)
        self.assertGreater(model_s.count_parameters(), 25_000_000)

        model_l = build_model("large", vocab_size=1000, init_weights=False)
        self.assertGreater(model_l.count_parameters(), 350_000_000)


if __name__ == "__main__":
    unittest.main()
