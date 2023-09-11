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


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_features(model, processor, frames, device):
    frame_chunks = chunks(frames, 32)
    features = []
    for chunk in frame_chunks:
        inputs = processor(images=chunk, return_tensors="pt").to(device)
        outputs = model(**inputs)
        features.extend(outputs.image_embeds.detach().cpu().numpy())
    features = np.array(features)
    return features


def load_frames(video_path, num_frames):
    vr = VideoReader(video_path)
    total_frames = len(vr)
    num_frames = total_frames # added this to get features for all frames
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
    
    for video in tqdm(test_videos):
        label = video.split('/')[0]
        video_id = video.split('/')[1].replace('.avi', '')
        if os.path.exists(os.path.join(save_folder, video_id + '.pkl')):
            continue
        
        frames = load_frames(os.path.join(videos_folder, video), num_frames)
        try:
            frame_features = get_features(model, processor, frames, device)
        except:
            print(f"error extracting features for video: {video_id} with shape {frames.shape}")
            continue
        out = {'frame_features': frame_features, 'label': label}
        pickle.dump(out, open(os.path.join(save_folder, video_id + '.pkl'), 'wb'))
       

def extract_label_features(save_folder, model_string):
    templates = [
        'a video of the person doing {}.',
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
        sub_action_texts = json.load(open(os.path.join(labels_folder, f), 'r'))['text'].split('\n')
        
        for template in templates[:1]:
            input_string = template.format(class_label)[:-1]
            inputs = [input_string + f" while {text[3:]}" for text in sub_action_texts]
            #inputs = [input_string]
            inputs = tokenizer(inputs, padding=True, return_tensors="pt").to(device)

            outputs = model(**inputs)
            features = outputs.text_embeds
            class_label = class_label.replace(' ', '')
            if class_label not in out:
                out[class_label] = []
            out[class_label].append(features.cpu().detach().numpy())

    return out


if __name__ == '__main__':
    model_string = "openai/clip-vit-base-patch16"
    save_folder = "./openai_clip_vit_base_patch16"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    label_features = extract_label_features(save_folder, model_string)
    extract_frame_features(save_folder, model_string)
    
    test_videos = [line.rstrip() for line in open('/home/c3-0/datasets/UCF101/testlist01.txt', 'r').readlines()]
    
    class_labels = list(label_features.keys())
    ground_truth, predictions = [], []
    for video in tqdm(test_videos):
        video_id = video.split('/')[1].replace('.avi', '')
        video_features = pickle.load(open(os.path.join(save_folder, video_id + '.pkl'), 'rb'))
        frame_features =  video_features[video_id]['frame_features']
        frame_features = torch.from_numpy(frame_features)
        video_label = video_features[video_id]['label']
        assert video_label.lower() in [label.lower() for label in class_labels]
        ground_truth.append([label.lower() for label in class_labels].index(video_label.lower()))
        scores = []
        for class_label in class_labels:
            text_features = label_features[class_label]
            text_features = np.array(text_features)
            num_templates, num_sub_actions = text_features.shape[0], text_features.shape[1]
            text_features = text_features.reshape(-1, text_features.shape[-1])
            text_features = torch.from_numpy(text_features)
            sim_matrx = (frame_features @ text_features.T).numpy()
            sim_matrx = sim_matrx.reshape(-1, num_templates, num_sub_actions)
            score = np.max(np.mean(np.mean(sim_matrx, axis=0), axis=0))
            scores.append(score)
        predictions.append(np.argmax(scores))

    accuracy = accuracy_score(ground_truth, predictions)
    print("accuracy: ", accuracy)
