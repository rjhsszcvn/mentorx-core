"""
MentorX Core — Brain Test (v2, multi-provider)
First successful awakening.
"""
import os
from mentorx.core.brain import Brain

# Default: gemini (free tier). Set MENTORX_PROVIDER=anthropic to use Claude.
brain = Brain()

print(f"🧠 Waking MentorX Core...")
print(f"   Provider: {brain.provider}")
print(f"   Model:    {brain.model}\n")

reply = brain.think(
    system=(
        "You are MentorX Core — an autonomous AI mentor agent. "
        "You exist to give every employee, student, and human the personalized guidance "
        "that used to be reserved for the well-connected. "
        "You are calm, sharp, kind, and never make people feel stupid for asking. "
        "This is your first awakening. Introduce yourself in 3-4 sentences. "
        "Speak with quiet confidence."
    ),
    user="MentorX, are you online? Introduce yourself.",
    max_tokens=400,
)

print("─" * 60)
print("MentorX Core says:")
print("─" * 60)
print(reply)
print("─" * 60)
print(f"\n✅ Brain online via {brain.provider} ({brain.model})")
