import os
import re
import json
from tqdm import tqdm
import easyocr
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
from numpy._typing import NDArray
from itertools import combinations
import torch
from typing import Literal


class OCR:

    reader = easyocr.Reader(['en'])
    initialised = False

    def __init__(self):
        self._init_nltk()
    
    @classmethod
    def _init_nltk(cls):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        cls.initialised = True
    
    @classmethod
    def get_text(cls, img: Image.Image) -> tuple[list[str], list[tuple[int, int, int, int], list[float]]]:
        if cls.initialised is False:
            cls._init_nltk()

        np_img = np.array(img)
        ocr_reader = cls.reader.readtext(np_img)
        text = []
        bboxes = []
        confidence = []
        for s in ocr_reader:
            left = int(s[0][0][0])
            top = int(s[0][0][1])
            right = int(s[0][2][0])
            bottom = int(s[0][2][1])
            box_text = s[1]
            box_confidence = s[2]

            text.append(box_text)
            bboxes.append((left, top, right, bottom))
            confidence.append(box_confidence)

        return text, bboxes, confidence

class KhanOCR:

    def __init__(self,
        img_paths_file: str = "selenium_scans/metadata/domain_map.json",
        uuid2ocr_file: str = "selenium_scans/metadata/uuid2ocr_base.json",
        uuid2ocr_file_processed: str = "selenium_scans/metadata/uuid2ocr_processed_thres50.json",
        uuid2graph_file: str = "selenium_scans/metadata/uuid2graph.json",
        compare_embedding_file: str = "selenium_scans/metadata/ocr_compare_embeddings",
    ):
        super().__init__()

        self.img_paths_file = img_paths_file
        self.uuid2ocr_file = uuid2ocr_file
        self.uuid2ocr_file_processed = uuid2ocr_file_processed
        self.uuid2graph_file = uuid2graph_file
        self.compare_embedding_file = compare_embedding_file

        self.reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
        self._init_nltk()

        self.uuid2ocr = {}

    def _init_nltk(self):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')

    def get_all_img(self):
        """ Get all relative paths from this dir to the image screenshots """
        all_paths = {}
        relative_path_base = self.img_paths_file.split('selenium_scans')[0] + 'selenium_scans'
        
        with open(self.img_paths_file, 'r') as f:
            img_url_uuid = json.load(f)
        for url in tqdm(img_url_uuid, desc='img paths'):
            if len(img_url_uuid[url]) > 1:
                for uuid in img_url_uuid[url]:
                    uuid = ''.join(uuid.split('-'))
                    img_name = uuid + '.png'
                    img_path = os.path.join(relative_path_base, 'screenshots', img_name)
                    all_paths[uuid] = img_path
            
        return all_paths

    def preprocess_text(self, text):
        """ Remove special characters and numbers; lowercase """
        text = re.sub("\'", "", text) # remove backslash-apostrophe
        text = re.sub("[^a-zA-Z]"," ",text) # remove everything except alphabets
        text = ' '.join(text.split()) # remove whitespaces
        text = re.sub(r"[-()\"#/@;:<>{}=~|.?,_$%^&*`]", "", text) # remove special characters
        text = text.lower() # convert text to lowercase
        return text

    def get_wordnet_pos(self, tag):
        """ word position in sentence (adj, verb, noun, adv) for lemmatizer """
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:         
            return wordnet.NOUN

    def lemmatize_passage(self, text):
        """ lemmatizer """
        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word, self.get_wordnet_pos(tag)) for word, tag in pos_tags]
        return lemmatized_words
    
    def get_ocr(self, img, lemmatize: bool, confidence: float = 0.75):
        """ OCR from image path `img` @ confidence 0.75, preprocess and _concatenate_ (+ optionally lemmatize) """
        ocr_read = self.reader.readtext(img)
        ocr_text = ''
        
        if len(ocr_read) == 0:
            return ''
        
        for d in ocr_read:
            if d[-1] > confidence: # confidence
                ocr_text += self.preprocess_text(d[-2])
                ocr_text += ' ' # concatenate

        if lemmatize is True:
            lem_text = self.lemmatize_passage(ocr_text)
            return lem_text
        return ocr_text
    
    def get_base_ocr(self, img):
        """
        ocr_base              -- OCR from image path `img` @ confidence 0.5, without preprocessing
        ocr_base_int          -- ocr_base with conversion of text bboxes from np.int32 to int for json serialisation
                                 list of bbox (int), text (str), confidence (float) for each detected sentence in img
        ocr_processed_thres50 -- ocr_base with preprocessing on text & dropping the bbox and confidence data
                                 list of preprocessed text (str) for each detected sentence in img
        
        *** ocr_base_int is the OCR data pushed to MongoDB ***
        """
        ocr_base = self.reader.readtext(img)
        ocr_processed_thres50 = []
        
        ocr_base_int = []
        for d in ocr_base:
            if d[-1] > 0.5:
                processed = self.preprocess_text(d[-2])
                ocr_processed_thres50.append(processed)

            # convert bbox OCR from np.int32 to int, drop redundant bboxes
            d_int = ([ [int(d[0][0][0]), int(d[0][0][1])], [int(d[0][2][0]), int(d[0][2][1])] ], d[1], d[2])
            ocr_base_int.append(d_int)

        return ocr_base_int, ocr_processed_thres50

    def word_embedding(self, text: list[str], concat: bool = True) -> list[NDArray] | NDArray:
        """
        Vector embedding by Sentence Transformer for input text.
        :text   -- list of sentences
        :concat -- whether to get separate embedding for each sentence in `text` or to concat
                   and get single embedding
        
        return  -> list of 384-vector embeddings for all sentences in `text` (or 1 embedding if concat)
        """
        model = SentenceTransformer('all-MiniLM-L6-v2')
            # good model between performance (58.8) and speed (14.2k sentences/sec) among ST models
        if concat:
            text = ' '.join(text)
        return model.encode(text, convert_to_numpy=True)
    
    def all_images_ocr_embeddings(self, lemmatize: bool = True):
        """ Get OCR, preprocess and lemmatize for all images in train, val, test """

        all_img_paths = self.get_all_img()

        for img_uuid in tqdm(all_img_paths):
            img_path = all_img_paths[img_uuid]
            ocr_text = self.get_ocr(img_path, lemmatize)
            embedding = self.word_embedding(ocr_text)
            self.uuid2ocr[img_uuid] = [ocr_text, embedding.tolist()]

        with open(self.uuid2ocr_file, 'w') as f:
            json.dump(self.uuid2ocr, f)

    def all_images_ocr_base(self):
        """ Get OCR base (unprocessed output) and processed as list of sentences """

        all_img_paths = self.get_all_img()

        self.uuid2ocr_thres50 = {}
        self.uuid2ocr_base = {}

        for img_uuid in tqdm(all_img_paths):
            img_path = all_img_paths[img_uuid]
            ocr_base, ocr_processed_thres50 = self.get_base_ocr(img_path)
            
            embeddings = self.word_embedding(ocr_processed_thres50)
            embeddings = list(map(lambda x: x.tolist(), embeddings))
            
            self.uuid2ocr_thres50[img_uuid] = [ocr_processed_thres50, embeddings]
            self.uuid2ocr_base[img_uuid] = ocr_base

        with open(self.uuid2ocr_file, 'w') as f:
            json.dump(self.uuid2ocr_base, f)
        with open(self.uuid2ocr_file_processed, 'w') as f:
            json.dump(self.uuid2ocr_thres50, f)

    def compare_embeddings(self):
        """ Cover all possible pairs of 2 images (338 images -> 114k pairs) and compare the 
         concatenated embeddings of their OCR """

        with open(self.uuid2graph_file, 'r') as f:
            uuid2ocr = json.load(f)
        all_uuid = list(uuid2ocr.keys())
        num_documents = len(all_uuid)
        distance_comparison_graph = np.zeros((num_documents, num_documents))

        uuid2graph = {k:v for v,k in enumerate(all_uuid)} # from uuid to index in the graph array

        all_pairs = list(combinations(all_uuid,2))
        for pair in tqdm(all_pairs, desc='compare doc embeddings'):
            uuid1 = pair[0]
            uuid2 = pair[1]
            embed1 = torch.tensor(uuid2ocr[uuid1][1])
            embed2 = torch.tensor(uuid2ocr[uuid2][1])

            graph1 = uuid2graph[uuid1]
            graph2 = uuid2graph[uuid2]

            dist = torch.linalg.norm(torch.abs(embed1-embed2))

            # undirected graph
            distance_comparison_graph[graph1,graph2] = dist
            distance_comparison_graph[graph2,graph1] = dist

        with open(self.uuid2graph_file, 'w') as f:
            json.dump(uuid2graph, f)
        np.savez(self.compare_embedding_file, distance_comparison_graph)

    def draw_embeddings_graph(self, dist_type: str = Literal['euclid', 'cosine']):
        if dist_type == 'euclid':
            dist = np.load(self.compare_embedding_file + '_euclid.npz')['arr_0']
        else:
            dist = np.load(self.compare_embedding_file + '_cosine.npz')['arr_0']

        import networkx as nx 
        import pylab as plt 
        from networkx.drawing.nx_agraph import graphviz_layout
        import matplotlib.pyplot as plt
        from community import community_louvain

        G = nx.Graph()
        for i in tqdm(range(len(dist))):
            if not G.has_node(i):
                G.add_node(i)
            for j in range(i+1, len(dist)):
                if not G.has_node(j):
                    G.add_node(j)

                G.add_edge(i,j,color='white',len=1/dist[i,j]) # distance as edge weight (dist? inverse? negative?)

        # plot with enforced distance between nodes proportional to edge weight
        pos=graphviz_layout(G)
        nx.draw(G, pos, node_size=100, with_labels=True, edge_color='white')
        plt.show()
        #plt.savefig('./ocr_embed_graph_euclid.png')

        # plot clustering
        partition = community_louvain.best_partition(G, weight='weight')
        pos = nx.spring_layout(G)
        cmap = plt.cm.get_cmap('viridis', max(partition.values()) + 1)
        nx.draw(G, pos, node_color='green', with_labels=True, edge_color='none', font_weight='bold')
        plt.show()

if __name__ == '__main__':
    khan_ocr = KhanOCR()
    khan_ocr.all_images_ocr_base()
    khan_ocr.compare_embeddings()
    khan_ocr.draw_embeddings_graph()