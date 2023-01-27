import torch
import torch.nn as nn
from typing import Any, List, Tuple, Union
from spacy.tokens.doc import Doc
from paragon.utils.lang_utils import POS_DICT_ as pos_dict
from paragon.utils.lang_utils import DEP_REL_DICT as dep_dict

def extract_dep_edge_info(dep: List) -> Tuple[List, List, List, List]:
    edge_index = [[], []]
    edge_tag = []
    node_pos = []
    node_phrases = []
    
    for node in dep:
        idx, phrase, pos_tag, head_idx, dep_tag = node
        node_pos_code = pos_dict[pos_tag]
        edge_index[0].append(idx)
        edge_index[0].append(head_idx)
        edge_index[1].append(head_idx)
        edge_index[1].append(idx)
        edge_tag.append(dep_dict[dep_tag])
        edge_tag.append(dep_dict[dep_tag])
        node_pos.append(node_pos_code)
        node_phrases.append(phrase)
    return node_pos, node_phrases, edge_index, edge_tag

class dep_tree_parser(nn.Module):
    '''
    transforms the language instructions into dependency tree
    '''
    def __init__(self) -> None:
        super().__init__()
        try:
            import spacy
        except ImportError as e:
            raise ImportError('Spacy backend requires the spaCy library. Install spaCy via pip first.') from e
        if spacy.__version__ < '3':
            model = 'en'
        else:
            model = 'en_core_web_trf'
        try:
            self.nlp = spacy.load(model)
        except OSError as e:
            raise ImportError('Unable to load the English model. Run `python -m spacy download en` first.') from e

    def process_list_spacy_doc(self, doc: List[Doc]) -> List:
        '''
        return a list of dependency tree (batch, dep_tree)
        '''
        data = map(self.process_spacy_doc, doc)
        return list(data) 

    def process_spacy_doc(self, doc: Doc) -> List:
        '''
        transform the spacy doc into list
        '''
        data = []
        for i in range(len(doc)):
            phrase = doc[i].text
            idx = doc[i].i
            dep_tag = doc[i].dep_
            pos_tag = doc[i].pos_
            head_idx = doc[i].head.i
            data.append([idx, phrase, pos_tag, head_idx, dep_tag])
        return data
            
    def forward(self, text: Union[str, List[str]]) -> List:
        '''
        Args:
            text: str, language sentence
        Returns:
            dependency_tree: list
        '''
        if isinstance(text, list):
            doc = list(self.nlp.pipe(text))
            data = self.process_list_spacy_doc(doc)
        else:
            doc = self.nlp(text)
            data = self.process_spacy_doc(doc)
        return data
