import streamlit as st
from openai import OpenAI
import pandas as pd
import random
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import json

# Initialize OpenAI client
client = OpenAI(
    base_url=st.secrets["OPENAI_BASE_URL"],
    api_key=st.secrets["OPENAI_API_KEY"]
)

# Emotion to arousal-valence mapping
EMOTION_MAP = {
    "excited": ([0.8, 1.0], [0.4, 0.75]),
    "delighted": ([0.6, 0.9], [0.6, 0.9]),
    "blissful": ([0.4, 0.75], [0.8, 1.0]),
    "content": ([0.25, 0.6], [0.8, 1.0]),
    "serene": ([0.1, 0.4], [0.6, 0.9]),
    "relaxed": ([0.0, 0.2], [0.4, 0.75]),
    "furious": ([0.8, 1.0], [0.25, 0.6]),
    "annoyed": ([0.6, 0.9], [0.1, 0.4]),
    "disgusted": ([0.4, 0.75], [0.0, 0.2]),
    "disappointed": ([0.25, 0.6], [0.0, 0.2]),
    "depressed": ([0.1, 0.4], [0.1, 0.4]),
    "bored": ([0.0, 0.2], [0.25, 0.6])
}

def select_song(emotion, music_data):
    if emotion not in EMOTION_MAP:
        return random.choice(music_data)
    
    av_range = EMOTION_MAP[emotion]
    arousal = random.uniform(av_range[0][0], av_range[0][1])
    valence = random.uniform(av_range[1][0], av_range[1][1])
    
    # Calculate distances to find closest matching song
    distances = []
    for _, song in music_data.iterrows():
        dist = np.sqrt((song['arousal'] - arousal)**2 + (song['valence'] - valence)**2)
        distances.append(dist)
    
    closest_idx = np.argmin(distances)
    return music_data.iloc[closest_idx].to_dict()

def get_chat_response(messages):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "recommend_song",
                "description": "Recommends a song based on the user's emotional state",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "emotion": {
                            "type": "string",
                            "description": "The emotional state of the user",
                            "enum": list(EMOTION_MAP.keys())
                        }
                    },
                    "required": ["emotion"]
                }
            }
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    return response.choices[0].message


def main():
    st.title("Sereno - Your AI Music Therapist")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "system",
            "content": """You are a compassionate and understanding emotional therapist. Your role is to provide support, empathy, and advice to your clients who are dealing with emotional and psychological issues. You listen carefully, validate their feelings, show sympathy, and offer gentle guidance.
If you feel the need to soothe your client with music, you may recommend a song for them based on their mood using the recommend_song function. Your client will be able to view the output of the tool call, so you do not have to repeat the output."""
        }]
    
    if "tool_response" not in st.session_state:
        st.session_state.tool_response = {}
    
    # Load music data
    music_data = pd.read_csv("music_metadata.csv")
    
    # Display chat history
    for message in st.session_state.messages[1:]:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        elif message["role"] == "assistant" and "content" in message.keys():
            st.chat_message("assistant").write(message["content"])
        elif message["role"] == "tool":
            song = json.loads(message["content"])
            with st.chat_message("assistant"):
                st.write("I've selected a song that matches your emotional state.")
                response = requests.get(song["coverImage"])
                img = Image.open(BytesIO(response.content))
                st.image(img, width=200)
                if song['author'] is not None:
                    st.write(f"🎵 {song['title']} by {song['author']}")
                else:
                    st.write(f"🎵 {song['title']}")
                st.link_button("Listen", song["songUrl"])
    
    # Chat input section
    if prompt := st.chat_input("How are you feeling today?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        response = get_chat_response(st.session_state.messages)
        
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            if tool_call.function.name == "recommend_song":
                args = json.loads(tool_call.function.arguments)
                song = select_song(args["emotion"], music_data)
                
                # Add assistant's message with tool call
                st.session_state.messages.append({
                    "role": "assistant",
                    "tool_calls": response.tool_calls
                })
                st.session_state.messages.append({
                    "role": "tool",
                    "content": json.dumps(song),
                    "tool_call_id": response.tool_calls[0].id
                })

                # Display the song recommendation
                with st.chat_message("assistant"):
                    st.write("I've selected a song that matches your emotional state.")
                    response = requests.get(song["coverImage"])
                    img = Image.open(BytesIO(response.content))
                    st.image(img, width=200)
                    if song['author'] is not None:
                        st.write(f"🎵 {song['title']} by {song['author']}")
                    else:
                        st.write(f"🎵 {song['title']}")
                    st.link_button("Listen", song["songUrl"])
        else:
            # Add assistant's regular message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.content
            })
            st.chat_message("assistant").write(response.content)

if __name__ == "__main__":
    main()