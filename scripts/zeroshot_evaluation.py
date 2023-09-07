from transformers import AutoProcessor, CLIPVisionModelWithProjection
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from sklearn.metrics import accuracy_score
import torch
import numpy as np
import os
from PIL import Image
import json
import pickle   
from tqdm import tqdm       
import decord 
decord.bridge.set_bridge('torch')
from decord import VideoReader


def get_features(model, processor, frames, device):
    inputs = processor(images=frames, return_tensors="pt").to(device)
    outputs = model(**inputs)
    features = outputs.image_embeds 
    return features.detach().cpu().numpy()


def load_frames(video_path, num_frames):
    vr = VideoReader(video_path)
    total_frames = len(vr)
    frame_ids = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = vr.get_batch(frame_ids)
    return frames


def get_labels(annotations, video_id, frame_ids, total_frames, num_classes=65):
    video_annotations = annotations[video_id]
    fps = total_frames/video_annotations['duration']
    labels = np.zeros((len(frame_ids), num_classes))
    for ann in video_annotations['actions']:
        action, start, end = ann
        start_frame, end_frame = int(start * fps), int(end * fps)
        indices = [indx for indx, frame_id in enumerate(frame_ids) if frame_id in range(start_frame, end_frame)]
        labels[indices, action - 1] = 1
    return labels


def extract_frame_features(save_folder, model_string):
    videos_folder = '/home/c3-0/datasets/UCF101/videos'
    num_frames = 32

    test_videos = [line.rstrip() for line in open('/home/c3-0/datasets/UCF101/testlist01.txt', 'r').readlines()]
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = CLIPVisionModelWithProjection.from_pretrained(model_string)
    processor = AutoProcessor.from_pretrained(model_string)

    model.to(device)
    
    out = {}
    for video in tqdm(test_videos):
        label = video.split('/')[0]
        video_id = video.split('/')[1].replace('.avi', '')
        frames = load_frames(os.path.join(videos_folder, video), num_frames)
        frame_features = get_features(model, processor, frames, device)
        out[video_id] = {}
        out[video_id]['frame_features'] = frame_features
        out[video_id]['label'] = label
    pickle.dump(out, open(os.path.join(save_folder, 'video_features.pkl'), 'wb'))


def extract_label_features(save_folder, model_string):
    templates = [
        'a photo of a person {}.',
        'a video of a person {}.',
        'a example of a person {}.',
        'a demonstration of a person {}.',
        'a photo of the person {}.',
        'a video of the person {}.',
        'a example of the person {}.',
        'a demonstration of the person {}.',
        'a photo of a person using {}.',
        'a video of a person using {}.',
        'a example of a person using {}.',
        'a demonstration of a person using {}.',
        'a photo of the person using {}.',
        'a video of the person using {}.',
        'a example of the person using {}.',
        'a demonstration of the person using {}.',
        'a photo of a person doing {}.',
        'a video of a person doing {}.',
        'a example of a person doing {}.',
        'a demonstration of a person doing {}.',
        'a photo of the person doing {}.',
        'a video of the person doing {}.',
        'a example of the person doing {}.',
        'a demonstration of the person doing {}.',
        'a photo of a person during {}.',
        'a video of a person during {}.',
        'a example of a person during {}.',
        'a demonstration of a person during {}.',
        'a photo of the person during {}.',
        'a video of the person during {}.',
        'a example of the person during {}.',
        'a demonstration of the person during {}.',
        'a photo of a person performing {}.',
        'a video of a person performing {}.',
        'a example of a person performing {}.',
        'a demonstration of a person performing {}.',
        'a photo of the person performing {}.',
        'a video of the person performing {}.',
        'a example of the person performing {}.',
        'a demonstration of the person performing {}.',
        'a photo of a person practicing {}.',
        'a video of a person practicing {}.',
        'a example of a person practicing {}.',
        'a demonstration of a person practicing {}.',
        'a photo of the person practicing {}.',
        'a video of the person practicing {}.',
        'a example of the person practicing {}.',
        'a demonstration of the person practicing {}.',
    ]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = CLIPTextModelWithProjection.from_pretrained(model_string)
    tokenizer = AutoTokenizer.from_pretrained(model_string)

    model.to(device)

    labels_folder = '/home/praveen/Research/ActionCLIP/data/direct_prompting_gpt3.5'
    out = {}
    for f in os.listdir(labels_folder):
        class_label = f.replace('.json', '')
        class_label = class_label.replace(' ', '')
        sub_action_texts = json.load(open(os.path.join(labels_folder, f), 'r'))['text'].split('\n')
        out[class_label] = []
        for template in templates:
            input_string = template.format(class_label)[:-1]
            inputs = [input_string + f" while {text}" for text in sub_action_texts]
            inputs = tokenizer(inputs, padding=True, return_tensors="pt").to(device)

            outputs = model(**inputs)
            features = outputs.text_embeds
            
            out[class_label].append(features.cpu().detach().numpy())

    pickle.dump(out, open(os.path.join(save_folder, 'label_features.pkl'), 'wb'))


if __name__ == '__main__':
    model_string = "openai/clip-vit-base-patch16"
    save_folder = "./openai_clip_vit_base_patch16"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    #extract_label_features(save_folder, model_string)
    #extract_frame_features(save_folder, model_string)
    
    video_features = pickle.load(open(os.path.join(save_folder, 'video_features.pkl'), 'rb'))
    label_features = pickle.load(open(os.path.join(save_folder, 'label_features.pkl'), 'rb'))
    class_labels = list(label_features.keys())
    ground_truth, predictions = [], []
    for video_id in tqdm(video_features):
        frame_features =  video_features[video_id]['frame_features']
        frame_features = torch.from_numpy(frame_features)
        video_label = video_features[video_id]['label']
        assert video_label.lower() in [label.lower() for label in class_labels]
        ground_truth.append([label.lower() for label in class_labels].index(video_label.lower()))
        scores = []
        for class_label in class_labels:
            text_features = label_features[class_label]
            text_features = np.array(text_features)
            #text_features = text_features[:, 0:1, :][:, None, :]
            num_templates, num_sub_actions = text_features.shape[0], text_features.shape[1]
            text_features = text_features.reshape(-1, text_features.shape[-1])
            text_features = torch.from_numpy(text_features)
            sim_matrx = (frame_features @ text_features.T).numpy()
            sim_matrx = sim_matrx.reshape(-1, num_templates, num_sub_actions)
            score = np.mean(np.mean(np.mean(sim_matrx, axis=0), axis=0))
            scores.append(score)
        predictions.append(np.argmax(scores))

    accuracy = accuracy_score(ground_truth, predictions)
    print("accuracy: ", accuracy)
