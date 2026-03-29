# SMART GLASSES FOR VISUALLY IMPAIRED INDIVIDUALS USING DEEP LEARNING

OVERVIEW

Visual impairment affects millions of people worldwide, significantly limiting their ability to independently navigate and interact with their surroundings. Everyday tasks such as walking through crowded environments, identifying objects, or avoiding obstacles become challenging and often require external assistance.

This project presents a low-cost, intelligent assistive solution in the form of smart glasses that leverage computer vision and deep learning to provide real-time auditory feedback. By converting visual information into meaningful spoken descriptions, the system enhances mobility, independence, and safety for visually impaired users.

THE PROBLEM

A large global population suffers from moderate to severe visual impairment. Despite advancements in assistive technologies, many existing solutions face critical limitations such as high cost, limited real-time performance, dependency on internet connectivity, and lack of contextual understanding.

Traditional tools such as white canes and guide dogs provide physical assistance but do not offer semantic understanding of the environment. They cannot identify objects, describe surroundings, or provide contextual awareness.

This creates a significant gap: there is a need for an affordable, real-time, and intelligent system capable of interpreting surroundings and communicating them effectively to the user.

PROPOSED SOLUTION

The proposed system is a wearable smart glasses solution powered by deep learning. It captures the user’s surroundings using a camera, detects objects in real time, and converts the detected information into structured audio feedback.

The system generates descriptive outputs such as:
“There is a chair to your left at a short distance.”

By transforming raw visual input into actionable auditory information, the system enables users to make safer and more informed decisions while navigating their environment.

KEY FEATURES

Real-time object detection using YOLO

Lightweight and portable wearable system

Instant audio feedback through text-to-speech

Structured scene description including object, position, and distance

Offline functionality without continuous internet requirement

Cost-effective alternative to existing assistive technologies

SYSTEM ARCHITECTURE

The system operates through the following pipeline:

Image Capture
A Raspberry Pi Camera continuously captures real-time video frames of the surroundings.

Object Detection
The YOLO (You Only Look Once) deep learning algorithm processes each frame to detect objects efficiently.

Text Generation
Detected objects are converted into structured sentences using predefined templates.

Audio Output
The generated text is converted into speech using a text-to-speech engine and delivered via earphones.

HARDWARE COMPONENTS

Raspberry Pi 4 (central processing unit)

Raspberry Pi Camera Module

Earphones or speakers for audio output

Battery pack for portable power supply

Connecting wires and wearable frame setup

SOFTWARE STACK

Python

OpenCV

YOLO (object detection model)

COCO dataset

pyttsx3 (text-to-speech engine)

IMPLEMENTATION HIGHLIGHTS

The system is designed to operate efficiently on resource-constrained hardware. The deep learning model is optimized to achieve real-time performance on the Raspberry Pi.

A low-latency processing pipeline ensures minimal delay between object detection and audio output. A template-based text generation approach is used to maintain clarity and speed, avoiding computationally expensive natural language models.

The design carefully balances performance, accuracy, and hardware limitations to deliver a practical and usable solution.

CHALLENGES FACED

Several challenges were encountered during development:

Limited computational power of Raspberry Pi for deep learning tasks

Difficulty in achieving smooth and natural speech output using lightweight TTS engines

Ensuring real-time performance without latency

Network and connection instabilities during development and testing

These challenges required optimization and trade-offs to ensure system reliability and usability.

FUTURE ENHANCEMENTS

The system can be further improved with the following features:

Integration of ultrasonic sensors for obstacle proximity detection

GPS-based navigation for outdoor assistance

Voice command support for hands-free interaction

Cloud integration for advanced processing capabilities

Improved natural language generation for more detailed descriptions

IMPACT

This project addresses a critical real-world accessibility problem by providing visually impaired individuals with real-time environmental awareness.

It enables independent navigation, reduces reliance on external assistance, and improves safety in daily activities. By making assistive technology more affordable and intelligent, the system has the potential to significantly enhance the quality of life for users.

CONCLUSION

The Smart Glasses system demonstrates the effective integration of deep learning and embedded systems to build impactful assistive technology.

Despite hardware limitations, the project successfully delivers a functional, real-time solution that provides meaningful assistance to visually impaired users. It serves as a strong foundation for future advancements in assistive AI, wearable computing, and human-centered design.

CONTRIBUTORS

Rohit Arun (VIT Vellore)
Nandhalal G Nair(VIT Vellore)

LICENSE

This project is intended for academic and research purposes.
