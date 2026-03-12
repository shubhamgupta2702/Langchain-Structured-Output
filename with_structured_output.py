from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id='meta-llama/Llama-2-7b-chat-hf',
  task='text-generation',
)

model = ChatHuggingFace(llm=llm)

class Review(TypedDict):
  key_themes:Annotated[list[str], "write all the key themes discussed in the review in a list"]
  summary:Annotated[str,"a brief summary of this review"]
  sentiment:Annotated[Literal['pos', 'neg'],'return sentiment of this review']
  pros:Annotated[Optional[list[str]], "Write down all the pros in the list"]
  cons:Annotated[Optional[list[str]], "Write down all the cons in the list"]
  
structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I've been using the Nexara Ultra 12 Pro Max for about six months now, and wow, what a ride it's been. Picked it up during the Black Friday sale for around $1,099 (128GB model in Midnight Black), and it's easily one of the best purchases I've made in years. Coming from an older flagship, this thing feels like jumping from a bicycle to a Ferrari – smooth, powerful, and packed with features that make daily life easier. But it's not perfect; there are some quirks that might turn off perfectionists. Let me break it down.

Design and Build Quality
Right out of the box, the premium vibe hits you. The titanium frame is lightweight yet sturdy at 215g, and the 6.8-inch curved AMOLED display with 144Hz refresh rate is buttery smooth for scrolling TikTok or gaming. IP68 water/dust resistance means I've accidentally dropped it in the sink twice without panic. The matte back resists fingerprints better than glossy rivals, but the camera bump is huge – it wobbles on flat surfaces unless you use a case (which they include a basic one). Buttons have great tactile feedback, and the in-display fingerprint scanner is lightning-fast, unlocking in under 0.2 seconds. Only gripe: no expandable storage, so you're stuck with internal space.

Performance and Software
Powered by the latest Quantum X4 chipset with 16GB RAM, this phone laughs at multitasking. I run 20+ Chrome tabs, edit 4K videos in Lightroom, and play Genshin Impact on ultra settings at 120fps without a single stutter or thermal throttle – even after an hour. Benchmarks? It crushes Geekbench at 2,800 single-core and 9,500 multi-core. NexaraOS 15 (based on Android 15) is clean and customizable, with AI features like real-time call transcription and photo object removal that actually work well. Three years of OS updates promised, which is solid. Battery optimization is top-notch; apps rarely drain in the background. Minor issue: occasional bloatware from carrier partners, but it's easy to uninstall.

Camera System
This is where it shines brightest – a 200MP main sensor with OIS, 50MP ultrawide, 12MP 5x telephoto, and a macro lens. Daytime shots are insanely detailed, with natural colors and zero oversharpening. Low-light performance is magical; night mode pulls in stars I didn't even see with my eyes. Portrait mode nails edge detection, and 8K video at 60fps is stabilized like a gimbal. Selfies from the 32MP front cam look great in all lighting. The AI editor lets you swap skies or remove photobombers effortlessly. Downsides? Zoom beyond 10x gets noisy, and video autofocus hunts in dim conditions. Still, it's pro-level for a phone.

Battery Life and Charging
The 5,500mAh battery is a champ – I get 8-10 hours of screen-on time with mixed use (social media, streaming, light gaming). Lasts a full day and then some, even with always-on display enabled. 80W wired charging hits 100% in 28 minutes, and 50W wireless is nearly as quick. Reverse wireless charging juices up my earbuds on the go. No charger in the box is annoying, though.

Audio, Haptics, and Extras
Stereo speakers are loud and bassy – perfect for YouTube without headphones. Haptics are precise, like typing on glass keys. 5G is blazing (up to 4.5Gbps on my carrier), Wi-Fi 7 is future-proof, and UWB chip works flawlessly with smart tags. Face unlock is secure and quick.

The Niggles and Value
Not all roses: software has minor bugs like occasional notification delays after updates, and the always-on display glitches in direct sunlight. No headphone jack (dongle needed), and eSIM-only might frustrate some. At full price ($1,299), it's steep, but sales make it a steal compared to competitors.""")

print(result)
print(result['summary'])
print(result['sentiment'])
