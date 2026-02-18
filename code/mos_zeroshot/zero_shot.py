
import os, random
import torch, torchaudio

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr

SUBSET_DATA = True
DATA_PARENT_DIR = '../ProbMOS/'
torch.hub.set_dir('./hub_chkpt/')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42); np.random.seed(42); torch.manual_seed(42)

##################################################################
# Utils

def subset_for_expts(label_df, n_subset=750):
    if SUBSET_DATA:
        n = len(label_df)
        return label_df.loc[
            np.random.choice(n, min(n, n_subset), replace=False)
        ].reset_index(drop=True)
    else:
        return label_df


def get_lm_decoder():
    from torchaudio.models.decoder import download_pretrained_files, ctc_decoder
    lm_files = download_pretrained_files("librispeech-4-gram")

    beam_search_decoder = ctc_decoder(
        lexicon=lm_files.lexicon,
        tokens=lm_files.tokens,
        lm=lm_files.lm,
        nbest=5,
        beam_size=1500,
        lm_weight=3.23,
        word_score=-0.26,
    )

    return beam_search_decoder


def load_non_english(dataset):
    if dataset == 'ood_22':
        labs = subset_for_expts(pd.concat([
            pd.read_csv('data/ood/DATA/sets/train_mos_list.txt', names=['file', 'mos']),
            pd.read_csv('data/ood/DATA/sets/val_mos_list.txt', names=['file', 'mos']),
            pd.read_csv('data/ood/DATA/sets/test_mos_list.txt', names=['file', 'mos']),
        ], axis=0).reset_index(drop=True))

        wav_path = 'data/ood/DATA/wav/'

    elif dataset == 'fr_23':
        labs = subset_for_expts(
            pd.read_csv('data/track1_answer.txt', names=['file', 'mos']),
        )
        labs.file += '.wav'

        wav_path = 'data/track1/VoiceMOS2023Track1/'

    elif dataset == 'sing_23':
        labs = subset_for_expts(
            pd.read_csv('data/track2_answer.txt', names=['file', 'mos']),
        )
        labs.file += '.wav'

        wav_path = 'data/track2/'

    elif dataset == 'ch_23':
        labs = pd.read_csv('data/track3_answer.txt', names=['file', 'mos'])
        labs.file += '.wav'
        wav_path = 'data/track3/'

        labs = labs.loc[[os.path.exists(os.path.join(wav_path, x)) for x in labs.file]].reset_index(drop=True)

        labs = subset_for_expts(labs)

    return labs, wav_path


class Main22Transcripts:
    def __init__(self):
        self.utt_map = pd.read_table('data/main/secret_utt_mappings.txt', names=['sec', 'utt'])

        transcripts = pd.read_table('main_track_truth_transcripts.txt', names=['file', 'transcript'])
        transcripts.loc[:, 'transcript'] = transcripts.loc[:, 'transcript'].str.lower()
        transcripts.loc[:, 'transcript'] = transcripts.loc[:, 'transcript'].str.replace('[,.:?`!;"_()]', '', regex=True)

        transcripts.loc[transcripts.transcript.str.contains('\r', regex=False), 'transcript'] = \
            transcripts.loc[transcripts.transcript.str.contains('\r', regex=False), 'transcript'].apply(lambda x: x.split('\r')[0])

        transcripts.loc[transcripts.transcript.str.contains('10'), 'transcript'] = \
            transcripts.loc[transcripts.transcript.str.contains('10'), 'transcript'].str.replace('10', 'ten')

        transcripts.loc[transcripts.transcript.str.contains('20'), 'transcript'] = \
            transcripts.loc[transcripts.transcript.str.contains('20'), 'transcript'].str.replace('20', 'twenty')

        self.transcripts = transcripts

    def get_it(self, file_name):
        file_name = self.utt_map.loc[self.utt_map.sec == file_name, 'utt'].iloc[0]
        return self.transcripts.loc[(self.transcripts.file + '.wav') == file_name, 'transcript'].iloc[0]


class ZeroShotModel:
    english_w2v_models = [
        'wav2vec_large.pt',
        'vq-wav2vec.pt',
    ]

    fairseq_w2v2_models = [
        'wav2vec_small.pt',
        'libri960_big.pt',
        'xlsr_53_56k.pt',
    ]

    english_w2v2_models = [
        'WAV2VEC2_BASE',
        'WAV2VEC2_LARGE',
        'WAV2VEC2_XLSR_2B',
        'WAV2VEC2_ASR_BASE_10M',
        'WAV2VEC2_ASR_BASE_100H',
        'WAV2VEC2_ASR_BASE_960H',
        'WAV2VEC2_ASR_LARGE_960H',
        'WAV2VEC2_ASR_LARGE_LV60K_960H',
        'VOXPOPULI_ASR_BASE_10K_EN',
        'HUBERT_ASR_LARGE',
        'HUBERT_ASR_XLARGE',
    ]

    multilingual_w2v2_models = [
        'WAV2VEC2_XLSR53',
        'VOXPOPULI_ASR_BASE_10K_FR',
    ]

    chinese_w2v2_models = [
        './chinese_base/',
        './chinese_large/',
    ]

    reductions = ['r_mean', 'r_sd', 'r_max', 'r_ent']

    def __init__(self, model_name):
        self.model_name = model_name
        self.preprocess_audio = False

        if (model_name in self.english_w2v_models) or (model_name in self.fairseq_w2v2_models):
            self._init_fairseq(model_name)
        elif model_name in self.english_w2v2_models:
            self._init_torchaudio(model_name)
        elif model_name in self.multilingual_w2v2_models:
            self._init_torchaudio(model_name)
        elif model_name in self.chinese_w2v2_models:
            self._init_huggingface(model_name)
        else:
            raise ValueError('Unknown model.')

        # white_noise = torch.randn(1, 10000).to(device)
        # white_noise /= white_noise.abs().max()
        # self.high_sig = self.compute_outputs(white_noise).mean(dim=0)

    def _init_fairseq(self, model_name, checkpoint_path='./fairseq/'):
        import fairseq
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([
            os.path.join(checkpoint_path, model_name)
        ])
        self.model = model[0].to(device).eval()

        if (model_name in self.english_w2v_models) and (not 'vq' in model_name):
            self.model.wav2vec_predictions.infonce = False
        if model_name in self.fairseq_w2v2_models:
            self.output_dim = 2
        else:
            self.output_dim = 11

    def _init_torchaudio(self, model_name):
        bundle = eval(f'torchaudio.pipelines.{model_name}')
        model_from_bundle = bundle.get_model().to(device).eval()

        self.output_dim = bundle._params['aux_num_out']
        if self.output_dim is None:
            self.output_dim = bundle._params['encoder_embed_dim']

        if hasattr(model_from_bundle, 'model'):
            self.preprocess_audio = True
            self.model = model_from_bundle.model
        else:
            self.model = model_from_bundle

    def _init_huggingface(self, model_name):
        from transformers import Wav2Vec2Model

        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model = self.model.to(device).eval()
        self.output_dim = 768 if 'base' in model_name else 1024

    def forward_chinese_dp(self, input, p_drop, n_mcd):
        with torch.no_grad():
            output = self.model.feature_extractor(input)
            output = output.transpose(1, 2)
            output = self.model.feature_projection(output)[0]

            output = output.repeat((1 if p_drop == 0.0 else n_mcd, 1, 1))

            output = torch.nn.functional.dropout(output, p_drop)
            output = self.model.encoder(output).last_hidden_state
        return output.mean(dim=0)

    def forward_dp(self, input, p_drop, n_mcd):
        with torch.no_grad():
            output = self.model.feature_extractor(input, None)[0]
            output = output.repeat((1 if p_drop == 0.0 else n_mcd, 1, 1))
            output = torch.nn.functional.dropout(output, p_drop)
            output = self.model.encoder(output)
            if self.model.aux is not None:
                output = self.model.aux(output)
        return output.mean(dim=0)

    def forward_fs(self, input, *args, **kwargs):
        if 'p_drop' in kwargs.keys():
            assert kwargs['p_drop'] == 0.0

        with torch.no_grad():
            output = self.model(input)['x'].squeeze()
            output = output.softmax(dim=0)
            assert output.shape[0] == 101
            output = torch.cat([output[[0]], output[1:, :].sum(axis=0)[None, ...]], axis=0).log()
        return output.T

    def compute_metrics_from_output(self, output):
        assert (len(output.shape) == 2) and (output.shape[1] == self.output_dim)

        # output - self.high_sig[None, :] # output[:, self.idx_to_choose]
        output[output == -np.inf] = output[output != -np.inf].min()

        results = [
            torch.mean(output, dim=1),
            torch.std(output, dim=1),
            torch.max(output, dim=1).values,
            torch.distributions.Categorical(probs=torch.softmax(output, dim=1)).entropy(),
        ]

        results = [r.mean().item() for r in results]

        return results

    def compute_outputs(self, audio, p_drop=0.0, n_mcd=100):
        input = audio.to(device)

        if len(input[0]) < 750:
            input = torch.cat([torch.zeros_like(input), input], axis=-1)

        if self.preprocess_audio:
            input = torch.nn.functional.layer_norm(input, input.shape)

        if self.model_name in self.english_w2v_models:
            output = self.model(input)['cpc_logits']
            output = output.reshape(-1, self.output_dim)
            # output = self.model.feature_extractor(input)
            # output = self.model.feature_aggregator(output).squeeze().T
        elif (self.model_name in self.english_w2v2_models) or (self.model_name in self.multilingual_w2v2_models):
            output = self.forward_dp(input, p_drop=p_drop, n_mcd=n_mcd)
        elif self.model_name in self.chinese_w2v2_models:
            output = self.forward_chinese_dp(input, p_drop=p_drop, n_mcd=n_mcd)
        elif self.model_name in self.fairseq_w2v2_models:
            output = self.forward_fs(input)

        return output.squeeze().double()

    def compute_featureset(self, audio, p_drop=0.0, n_mcd=100):
        output = self.compute_outputs(audio, p_drop=p_drop, n_mcd=n_mcd)
        return self.compute_metrics_from_output(output)

    def __call__(self, audio_list, **kwargs):
        return np.array([self.compute_featureset(x, **kwargs) for x in tqdm(audio_list, leave=False)])

    @staticmethod
    def get_correlations(outputs, target):
        return np.round(spearmanr(outputs, target).statistic, 3)[-1, :-1]


if __name__ == '__main__':

    os.chdir(DATA_PARENT_DIR)

    ################################################################
    # ----------------------- Expt 0: noise forwards
    ### what does running a random audio sequence through these models output?

    # outputs = {}

    # for model_name in ZeroShotModel.english_w2v_models + ZeroShotModel.english_w2v2_models + ZeroShotModel.chinese_w2v2_models:
    #     model = ZeroShotModel(model_name)

    #     for distb in [torch.randn, torch.rand]:
    #         for output_dim in [2000, 4000, 10000]:
    #             for dp in [0.0, 0.6]:
    #                 audio = distb(1, output_dim)
    #                 audio /= audio.abs().max()

    #                 out = model.compute_outputs(audio.to(device), p_drop=dp).detach().cpu()
    #                 assert len(out.shape) == 2
    #                 outputs[f'n:{distb == torch.randn}|dim:{output_dim}|{model_name}|dp:{dp}'] = out
    #                 print(f'n:{distb == torch.randn}|dim:{output_dim}|{model_name}|dp:{dp}')

    # torch.save(outputs, 'noise_images.pth')

    ################################################################
    # ----------------------- Expt 1: librosa reconstructed audio
    ### can UMs recover the ordering of which audio are more affected by compression

    import librosa

    audio = torchaudio.load('data/main/DATA/wav/sys92962-utt6d3c80e.wav')[0][0].numpy()
    model = ZeroShotModel('wav2vec_large.pt')

    mfcc_list = [1, 2, 4, 8, 32, 64, 128]
    reconstructed_audio_list = []
    for n_mfcc in mfcc_list:
        mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=n_mfcc)
        reconstructed_audio = librosa.feature.inverse.mfcc_to_audio(mfcc)
        reconstructed_audio = torch.tensor(reconstructed_audio)[None, ...].to(device)
        reconstructed_audio_list.append(reconstructed_audio)

    outputs = model(reconstructed_audio_list)
    print(f'mfcc_expt:{model.get_correlations(outputs, mfcc_list)}')

    ################################################################
    # ----------------------- load main 22 data

    labs = subset_for_expts(pd.concat([
        pd.read_csv('data/main/DATA/sets/train_mos_list.txt', names=['file', 'mos']),
        pd.read_csv('data/main/DATA/sets/val_mos_list.txt', names=['file', 'mos']),
        pd.read_csv('data/main/DATA/sets/test_mos_list.txt', names=['file', 'mos']),
    ], axis=0).reset_index(drop=True))

    # ----------------------- Expt 2: main 2022: w2v

    for model_name in ZeroShotModel.english_w2v_models:
        model = ZeroShotModel(model_name)

        audio_list = []
        for file in list(labs.file):
            x, sr = torchaudio.load(os.path.join('data/main/DATA/wav/', file))
            x = torchaudio.transforms.Resample(orig_freq=sr)(x)
            audio_list.append(x)

        outputs = model(audio_list)
        print(f'main22_expt_{model_name}:{model.get_correlations(outputs, labs.mos)}')

    # ----------------------- Expt 3: main 2022: w2v2

    for model_name in ['WAV2VEC2_XLSR53'] + ZeroShotModel.english_w2v2_models:
        model = ZeroShotModel(model_name)

        audio_list = []
        for file in list(labs.file):
            x, sr = torchaudio.load(os.path.join('data/main/DATA/wav/', file))
            x = torchaudio.transforms.Resample(orig_freq=sr)(x)
            audio_list.append(x)

        outputs = model(audio_list)
        print(f'main22_expt_{model_name}:{model.get_correlations(outputs, labs.mos)}')

    def my_compute_preds(self, x, y, negatives):

        from IPython.core.debugger import set_trace; set_trace()

        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1)
        logits = logits / self.logit_temp
        logits = logits.type_as(x)

        if is_xla_tensor(logits) or neg_is_pos.any():
            if not hasattr(self, "_inftensor"):
                fillval = -float(2**30)
                self._inftensor = (
                    torch.tensor(fillval).to(x.device)
                    if is_xla_tensor(logits)
                    else float("-inf")
                )
            logits[1:] = index_put(logits[1:], neg_is_pos, self._inftensor)

        return logits

    for model_name in ZeroShotModel.fairseq_w2v2_models:
        model = ZeroShotModel(model_name)

        audio_list = []
        for file in list(labs.file):
            x, sr = torchaudio.load(os.path.join('data/main/DATA/wav/', file))
            x = torchaudio.transforms.Resample(orig_freq=sr)(x)
            audio_list.append(x)

        outputs = model(audio_list)
        print(f'main22_expt_{model_name}:{model.get_correlations(outputs, labs.mos)}')

    # ----------------------- Expt 4: main 2022: w2v2 mc dropout

    model = ZeroShotModel('WAV2VEC2_ASR_BASE_960H')
    audio_list = []
    for file in list(labs.file):
        x, sr = torchaudio.load(os.path.join('data/main/DATA/wav/', file))
        x = torchaudio.transforms.Resample(orig_freq=sr)(x)
        audio_list.append(x)

    for dp in np.linspace(0.0, 1.0, 20):
        outputs = model(audio_list, p_drop=dp)
        print(f'dp_expt_{np.round(dp, 2)}:{abs(model.get_correlations(outputs, labs.mos)).mean()}')

    # ----------------------- Expt 5: main 2022: w2v2 legibility

    model = ZeroShotModel('WAV2VEC2_ASR_BASE_960H')
    beam_search_decoder = get_lm_decoder()
    transcripts = Main22Transcripts()

    results = []
    for i, file_name in enumerate(bar := tqdm(list(labs.file))):
        waveform, sample_rate = torchaudio.load(os.path.join('data/main/DATA/wav/', file_name))
        waveform = waveform.to(device)

        emission = model.forward_dp(waveform, p_drop=0.85, n_mcd=100)

        beam_search_result = beam_search_decoder(emission[None, ...].cpu())
        beam_search_transcript = " ".join(beam_search_result[0][0].words).strip()

        try:
            actual_transcript = transcripts.get_it(file_name)

            beam_search_wer = torchaudio.functional.edit_distance(
                actual_transcript.split(),
                beam_search_result[0][0].words
            ) / len(actual_transcript)

            results.append([
                beam_search_wer,
                labs.loc[labs.file == file_name, 'mos'].mean(),
            ] + model.compute_metrics_from_output(emission))

            if i > 7:
                res_df = pd.DataFrame(results)
                res_df.columns = ['WER', 'MOS'] + model.reductions
                r = spearmanr(res_df).statistic
                bar.set_description(f'R->WER:{np.round(abs(r[2:, 0]).mean(), 2)}; R->MOS:{np.round(abs(r[2:, 1]).mean(), 2)}; WER->MOS:{np.round(r[0, 1], 2)}')
        except IndexError as e:
            pass

    # R->WER:0.7; R->MOS:0.6; WER->MOS:-0.53 (dp 0.8)

    ################################################################
    # ----------------------- Expt 6: non-eng: w2v

    for model_name in ZeroShotModel.english_w2v_models:
        model = ZeroShotModel(model_name)

        for dataset in ['ood_22', 'fr_23', 'sing_23', 'ch_23']:
            labs, wav_path = load_non_english(dataset)

            audio_list = []
            for file in list(labs.file):
                x, sr = torchaudio.load(os.path.join(wav_path, file))
                x = torchaudio.transforms.Resample(orig_freq=sr)(x)
                audio_list.append(x)

            outputs = model(audio_list)

            if dataset == 'fr_23':
                idx_ad = list(labs.file.str.contains('AD_test'))

                ad_out = [x for (i, x) in enumerate(outputs) if idx_ad[i]]
                neb_out = [x for (i, x) in enumerate(outputs) if not idx_ad[i]]

                print(f'{dataset + ":ad"}_expt_{model_name}:{model.get_correlations(ad_out, labs.loc[idx_ad, "mos"])}')
                print(f'{dataset + ":neb"}_expt_{model_name}:{model.get_correlations(neb_out, labs.loc[[not x for x in idx_ad], "mos"])}')
            else:
                print(f'{dataset}_expt_{model_name}:{model.get_correlations(outputs, labs.mos)}')

    ################################################################
    # ----------------------- Expt 7: non-eng: w2v2

    for dataset in ['fr_23', 'sing_23', 'ch_23', 'ood_22']:
        labs, wav_path = load_non_english(dataset)

        if dataset in ['ood_22', 'ch_23']:
            model_name = 'WAV2VEC2_XLSR53'
        elif dataset == 'sing_23':
            model_name = 'WAV2VEC2_LARGE'
        elif dataset == 'fr_23':
            model_name = 'VOXPOPULI_ASR_BASE_10K_FR'
            # voiced_idx = np.argwhere([(x not in ['-', '|', "'"]) and ("<" not in x) for x in bundle.get_labels()])[:, 0]
        else:
            raise ValueError

        model = ZeroShotModel(model_name)

        audio_list = []
        for file in list(labs.file):
            x, sr = torchaudio.load(os.path.join(wav_path, file))
            x = torchaudio.transforms.Resample(orig_freq=sr)(x)
            audio_list.append(x)

        for dp in [0.0, 0.6, 0.8]:
            outputs = model(audio_list, p_drop=dp, n_mcd=50)

            if dataset == 'fr_23':
                idx_ad = list(labs.file.str.contains('AD_test'))
                ad_out = [x for (i, x) in enumerate(outputs) if idx_ad[i]]
                neb_out = [x for (i, x) in enumerate(outputs) if not idx_ad[i]]
                print(f'{dataset + ":ad"}_expt_{model_name}_dp{dp}:{model.get_correlations(ad_out, labs.loc[idx_ad, "mos"])}')
                print(f'{dataset + ":neb"}_expt_{model_name}_dp{dp}:{model.get_correlations(neb_out, labs.loc[[not x for x in idx_ad], "mos"])}')
            else:
                print(f'{dataset}_expt_{model_name}_dp{dp}:{model.get_correlations(outputs, labs.mos)}')

    ################################################################
    # ----------------------- Expt 8: chinese w2v2

    # the modele here are from github: TencentGameMate/chinese_speech_pretrain
    for dataset in ['ch_23', 'ood_22']:
        labs, wav_path = load_non_english(dataset)

        for model_name in ZeroShotModel.chinese_w2v2_models:
            model = ZeroShotModel(model_name)

            audio_list = []
            for file in list(labs.file):
                x, sr = torchaudio.load(os.path.join(wav_path, file))
                x = torchaudio.transforms.Resample(orig_freq=sr)(x)
                audio_list.append(x)

            for dp in [0.0, 0.6, 0.8]:
                outputs = model(audio_list, p_drop=dp, n_mcd=50)
                print(f'{dataset}_expt_{model_name}_dp{dp}:{model.get_correlations(outputs, labs.mos)}')

    ################################################################
    # ----------------------- Expt 8: ch w2v finetuned

    from tqdm import trange
    import wandb

    model = ZeroShotModel('wav2vec_large.pt')

    ################
    # Data

    labs, wav_path = load_non_english('ch_23')  # fr_23 # labs = labs.loc[labs.file.str.contains('NEB'), :].reset_index(drop=True)

    # from datasets import load_dataset
    # cv_11 = load_dataset("mozilla-foundation/common_voice_11_0", "zh-CN", split="test")

    # training_data = []
    # for i in trange(len(cv_11)):
    #     audio = torch.tensor(cv_11[i]['audio']['array']).float()[None, ...]
    #     sr = cv_11[i]['audio']['sampling_rate']

    #     if sr != 16000:
    #         audio = torchaudio.functional.resample(audio, sr, 16000)
    #     if audio.abs().max() > 1:
    #         audio /= audio.abs().max()
    #     if len(audio[0]) > 2000 and len(audio[0]) <= 160000:
    #         training_data.append(audio.to(device))

    training_data = []  # evdt
    for file in list(labs.file):  # labs.loc[labs.mos > 3.5, 'file']
        audio, sr = torchaudio.load(os.path.join(wav_path, file))

        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        if audio.abs().max() > 1:
            audio /= audio.abs().max()
        if len(audio[0]) > 2000 and len(audio[0]) <= 160000:
            training_data.append(audio.to(device))

    # SUBSET_DATA = True
    # labs = subset_for_expts(labs, n_subset=100)

    # training_data = []  # gpc
    # for root, dirs, files in tqdm(os.walk('/Database/ELRA_GlobalPhoneCorpus/Mandarin/wav/')):
    #     for file in files:
    #         if file.endswith(".wav"):
    #             try:
    #                 audio, sr = torchaudio.load(os.path.join(root, file))
    #                 if audio.numel() > 0:
    #                     if sr != 16000:
    #                         audio = torchaudio.functional.resample(audio, sr, 16000)
    #                     if audio.abs().max() > 1:
    #                         audio /= audio.abs().max()
    #                     if len(audio[0]) > 2000 and len(audio[0]) <= 160000:
    #                         training_data.append(audio.to(device))
    #             except:
    #                 pass

    # training_data = [training_data[i] for i in np.random.choice(len(training_data), 500, replace=False)]

    valid_audio_list = []
    for file in list(labs.file):
        x, sr = torchaudio.load(os.path.join(wav_path, file))
        x = torchaudio.transforms.Resample(orig_freq=sr)(x)
        valid_audio_list.append(x)

    ################
    # Finetune

    optimizer = torch.optim.Adam(model.model.parameters(), lr=5e-5); optim_steps=0

    wandb.init(entity='ml-at-cl', project='probmos')
    while True:

        # assert not model.model.wav2vec_predictions.infonce

        optimizer.zero_grad()
        loss = 0.0

        for i in np.random.choice(len(training_data), 25, replace=False):

            audio = training_data[int(i)]
            out = model.model(audio)
            loss += torch.nn.functional.binary_cross_entropy_with_logits(
                out['cpc_logits'],
                out['cpc_targets']
            )

        loss.backward()
        optimizer.step()

        if optim_steps % 25 == 0:
            model.model = model.model.eval()

            valid_outputs = model(valid_audio_list)
            v = abs(model.get_correlations(valid_outputs, labs.mos)).max()

            model.model = model.model.train()
            print(v)

        optim_steps += 1
        wandb.log(dict(loss=loss.item(), v=v))

    torch.save(model.model.cpu().state_dict(), 'ch_w2v_ft.pth')
