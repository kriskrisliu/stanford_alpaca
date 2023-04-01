import os,json
import fire
from tqdm import tqdm
from fastparquet import ParquetFile

def main(
        indir, 
        outdir='./', 
        update_blip=False,
        ):

    content_list = []
    pp_list = [pp for pp in os.listdir(indir) if pp.endswith('.parquet')]
    for pp in tqdm(pp_list):
        path = os.path.join(indir, pp)
        pf = ParquetFile(path)
        dF = pf.to_pandas()
        dF = dF[dF.similarity<1.]
        dF = dF[dF.similarity>0.33]
        dF = dF[dF.TEXT.str.len()>150]
        for idx, row in dF.iterrows():
            text_dict = {
                        'input': row.TEXT[:len(row.TEXT)//2],
                        'output': row.TEXT,
                        }
            content_list.append(text_dict)

    with open(os.path.join(outdir,f"tcontinue_data.json"),"w") as fp:
        json.dump(content_list, fp, indent=4)


if __name__=="__main__":
    fire.Fire(main)