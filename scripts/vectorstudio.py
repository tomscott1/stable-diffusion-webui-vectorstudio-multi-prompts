"""
using POTRACE as backend cmd line tool for vectorizing SD output
This script will download from

https://potrace.sourceforge.net/#downloading

the windows exetuable (todo: mac, linux support)
Potrace is under GPL, you can download the source from the url above.

If you dont want to download that, please install POTRACE to your 
system manually and assign it to your PATH env variable properly.
"""

# not yet
BASE_PROMPT=",(((lineart))),((low detail)),(simple),high contrast,sharp,2 bit"
BASE_NEGPROMPT="(((text))),((color)),(shading),background,noise,dithering,gradient,detailed,out of frame,ugly,error,Illustration, watermark"

BASE_STEPS=40
BASE_SCALE=10

StyleDict = {
    "Illustration":BASE_PROMPT+",(((vector graphic))),medium detail",
    "Logo":BASE_PROMPT+",(((centered vector graphic logo))),negative space,stencil,trending on dribbble",
    "Drawing":BASE_PROMPT+",(((cartoon graphic))),childrens book,lineart,negative space",
    "Artistic":BASE_PROMPT+",(((artistic monochrome painting))),precise lineart,negative space",
    "Tattoo":BASE_PROMPT+",(((tattoo template, ink on paper))),uniform lighting,lineart,negative space",
    "Gothic":BASE_PROMPT+",(((gothic ink on paper))),H.P. Lovecraft,Arthur Rackham",
    "Anime":BASE_PROMPT+",(((clean ink anime illustration))),Studio Ghibli,Makoto Shinkai,Hayao Miyazaki,Audrey Kawasaki",
    "Cartoon":BASE_PROMPT+",(((clean ink funny comic cartoon illustration)))",
    "Sticker":",(Die-cut sticker, kawaii sticker,contrasting background, illustration minimalism, vector, pastel colors)",
    "Gold Pendant": ",gold dia de los muertos pendant, intricate 2d vector geometric, cutout shape pendant, blueprint frame lines sharp edges, svg vector style, product studio shoot",
    "None - prompt only":""
}

##########################################################################
# prompts_from_file.py imports
import copy
import random
import shlex

import os
import platform
import shutil
import tempfile
import zipfile
import tarfile
import requests

import pathlib
import subprocess

import requests
import os.path

import modules.scripts as scripts
import gradio as gr

from modules.processing import Processed, process_images
from modules.shared import opts, state
from modules import script_callbacks, scripts, shared, sd_models

usefulDirs = scripts.basedir().split(os.sep)[-2:]
iframesrc        = "file="+usefulDirs[0]+"/"+usefulDirs[1]+"/scripts/editor/iife-index.html"
iframesrcSVGCODE = "file="+usefulDirs[0]+"/"+usefulDirs[1]+"/scripts/svgcode/index.html"
script_list_component = None

def check_ext(ext):
    found = False
    scripts_list = scripts.list_scripts("scripts", ".py")
    for scriptfile in scripts_list:
            if ext in scriptfile.basedir.lower():
                found = True
                break
    return found

# start prompts_from_file.py functions
def process_model_tag(tag):
    info = sd_models.get_closet_checkpoint_match(tag)
    assert info is not None, f'Unknown checkpoint: {tag}'
    return info.name


def process_string_tag(tag):
    return tag


def process_int_tag(tag):
    return int(tag)


def process_float_tag(tag):
    return float(tag)


def process_boolean_tag(tag):
    return True if (tag == "true") else False


prompt_tags = {
    "sd_model": process_model_tag,
    "outpath_samples": process_string_tag,
    "outpath_grids": process_string_tag,
    "prompt_for_display": process_string_tag,
    "prompt": process_string_tag,
    "negative_prompt": process_string_tag,
    "styles": process_string_tag,
    "seed": process_int_tag,
    "subseed_strength": process_float_tag,
    "subseed": process_int_tag,
    "seed_resize_from_h": process_int_tag,
    "seed_resize_from_w": process_int_tag,
    "sampler_index": process_int_tag,
    "sampler_name": process_string_tag,
    "batch_size": process_int_tag,
    "n_iter": process_int_tag,
    "steps": process_int_tag,
    "cfg_scale": process_float_tag,
    "width": process_int_tag,
    "height": process_int_tag,
    "restore_faces": process_boolean_tag,
    "tiling": process_boolean_tag,
    "do_not_save_samples": process_boolean_tag,
    "do_not_save_grid": process_boolean_tag
}


def cmdargs(line):
    args = shlex.split(line)
    pos = 0
    res = {}

    while pos < len(args):
        arg = args[pos]

        assert arg.startswith("--"), f'must start with "--": {arg}'
        assert pos+1 < len(args), f'missing argument for command line option {arg}'

        tag = arg[2:]

        if tag == "prompt" or tag == "negative_prompt":
            pos += 1
            prompt = args[pos]
            pos += 1
            while pos < len(args) and not args[pos].startswith("--"):
                prompt += " "
                prompt += args[pos]
                pos += 1
            res[tag] = prompt
            continue


        func = prompt_tags.get(tag, None)
        assert func, f'unknown commandline option: {arg}'

        val = args[pos+1]
        if tag == "sampler_name":
            val = sd_samplers.samplers_map.get(val.lower(), None)

        res[tag] = func(val)

        pos += 2

    return res


def load_prompt_file(file):
    if file is None:
        return None, gr.update(), gr.update(lines=7)
    else:
        lines = [x.strip() for x in file.decode('utf8', errors='ignore').split("\n")]
        return None, "\n".join(lines), gr.update(lines=7)
# end prompts_from_file.py functions
    
class Script(scripts.Script):
    def title(self):
        # return "Vector Studio"
        # return "Prompts from file or textbox"
        return "Vector Studio and Multi Prompts"

    def show(self, is_img2img):
        return True # scripts.AlwaysVisible

    def ui(self, is_img2img):
        self.run_callback = False

        # start prompts_from_file.py
        checkbox_iterate = gr.Checkbox(label="Iterate seed every line", value=False, elem_id=self.elem_id("checkbox_iterate"))
        checkbox_iterate_batch = gr.Checkbox(label="Use same random seed for all lines", value=False, elem_id=self.elem_id("checkbox_iterate_batch"))
        prompt_position = gr.Radio(["start", "end"], label="Insert prompts at the", elem_id=self.elem_id("prompt_position"), value="start")

        prompt_txt = gr.Textbox(label="List of prompt inputs", lines=1, elem_id=self.elem_id("prompt_txt"))
        file = gr.File(label="Upload prompt inputs", type='binary', elem_id=self.elem_id("file"))

        file.change(fn=load_prompt_file, inputs=[file], outputs=[file, prompt_txt, prompt_txt], show_progress=False)

        # We start at one line. When the text changes, we jump to seven lines, or two lines if no \n.
        # We don't shrink back to 1, because that causes the control to ignore [enter], and it may
        # be unclear to the user that shift-enter is needed.
        prompt_txt.change(lambda tb: gr.update(lines=7) if ("\n" in tb) else gr.update(lines=2), inputs=[prompt_txt], outputs=[prompt_txt], show_progress=False)
        # end prompts_from_file.py

        with gr.Group():
            with gr.Accordion("Vector Studio", open=False):

                with gr.Row():
                    poUseColor = gr.Radio(list(StyleDict.keys()), label="Visual style", value="Illustration")

                with gr.Row():

                    with gr.Column():
                            with gr.Box():
                                with gr.Group():
                                    with gr.Row():
                                        poDoVector = gr.Checkbox(label="Enable Vectorizing", value=True)
                                        poFormat = gr.Dropdown(["svg","pdf"], label="Output format", value="svg")
                                        poOpaque = gr.Checkbox(label="White is Opaque", value=True)
                                        poTight = gr.Checkbox(label="Cut white margin from input", value=True)
                                    with gr.Row():
                                        poKeepPnm = gr.Checkbox(label="Keep temp images", value=False)
                                        poThreshold = gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.5)

                    with gr.Column():
                            with gr.Box():
                                with gr.Group():
                                    poTransPNG      = gr.Checkbox(label="Transparent PNG",value=False)
                                    poTransPNGEps   = gr.Slider(label="Noise Tolerance",minimum=0,maximum=128,value=16)
                                    poTransPNGQuant = gr.Slider(label="Quantize",minimum=2,maximum=255,value=16)

                # return [poUseColor,poFormat, poOpaque, poTight, poKeepPnm, poThreshold, poTransPNG, poTransPNGEps,poDoVector,poTransPNGQuant]
                # return [checkbox_iterate, checkbox_iterate_batch, prompt_position, prompt_txt]
                return [checkbox_iterate, checkbox_iterate_batch, prompt_position, prompt_txt, poUseColor,poFormat, poOpaque,
                        poTight, poKeepPnm, poThreshold, poTransPNG, poTransPNGEps,poDoVector,poTransPNGQuant]


    def check_and_install_potrace(self):
        # Get the appropriate download URL based on the OS
        if platform.system() == 'Windows':
            download_url = "https://potrace.sourceforge.net/download/1.16/potrace-1.16.win64.zip"
            extension = '.zip'
            PO_TO_CALL = usefulDirs[0]+"/"+usefulDirs[1]+"/bin/potrace.exe"
        elif platform.system() == 'Linux':
            download_url = "https://potrace.sourceforge.net/download/1.16/potrace-1.16.linux-x86_64.tar.gz"
            extension = '.tar.gz'
            PO_TO_CALL = usefulDirs[0]+"/"+usefulDirs[1]+"/bin/potrace"
        elif platform.system() == 'Darwin':
            download_url = "https://potrace.sourceforge.net/download/1.16/potrace-1.16.mac-x86_64.tar.gz"
            extension = '.tar.gz'
            PO_TO_CALL = usefulDirs[0]+"/"+usefulDirs[1]+"/bin/potrace"
        else:
            raise ValueError("Unsupported operating system.")

        # Check if the PO_TO_CALL executable exists
        if not os.path.exists(PO_TO_CALL):

            print ("VectorStudio-Extension: Cannot find any POTRACE-executable in this installation.")
            print (f"VectorStudio-Extension: downloading from {download_url}... ")

            # Download the appropriate package
            response = requests.get(download_url)
            if response.status_code != 200:
                raise Exception("Failed to download Potrace package.")

            # Create a temporary directory to extract the package
            temp_dir = tempfile.mkdtemp()

            print (f"VectorStudio-Extension: extracting POTRACE executable... ")

            try:
                # Save the downloaded package to a temporary file
                temp_file = os.path.join(temp_dir, f"potrace_package{extension}")
                with open(temp_file, 'wb') as file:
                    file.write(response.content)

                # Extract the package based on the file extension
                if extension == '.zip':
                    with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                elif extension == '.tar.gz':
                    with tarfile.open(temp_file, 'r:gz') as tar_ref:
                        tar_ref.extractall(temp_dir)

                # Find the potrace executable file
                executable = None
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.startswith('potrace') and (file.endswith('.exe') or file.endswith('')):
                            executable = os.path.join(root, file)
                            break

                if executable is None:
                    raise Exception("Failed to find the potrace executable in the downloaded package.")

                # Move the potrace executable to the specified bin folder
                shutil.move(executable, PO_TO_CALL)
                print (f"VectorStudio-Extension: POTRACE executable successfully installed. ")

            finally:
                # Clean up the temporary directory
                shutil.rmtree(temp_dir)
        return PO_TO_CALL


    def run(self, p, checkbox_iterate, checkbox_iterate_batch, prompt_position, prompt_txt: str, 
            poUseColor, poFormat, poOpaque, poTight, poKeepPnm, poThreshold, poTransPNG,
            poTransPNGEps,poDoVector, poTransPNGQuant):
        ######################################################################################
        # start prompts_from_file.py
        lines = [x for x in (x.strip() for x in prompt_txt.splitlines()) if x]

        p.do_not_save_grid = True

        job_count = 0
        jobs = []

        for line in lines:
            if "--" in line:
                try:
                    args = cmdargs(line)
                except Exception:
                    errors.report(f"Error parsing line {line} as commandline", exc_info=True)
                    args = {"prompt": line}
            else:
                args = {"prompt": line}

            job_count += args.get("n_iter", p.n_iter)

            jobs.append(args)

        print(f"Will process {len(lines)} lines in {job_count} jobs.")
        if (checkbox_iterate or checkbox_iterate_batch) and p.seed == -1:
            p.seed = int(random.randrange(4294967294))

        state.job_count = job_count

        images = []
        all_prompts = []
        infotexts = []
        for args in jobs:
            state.job = f"{state.job_no + 1} out of {state.job_count}"

            copy_p = copy.copy(p)
            for k, v in args.items():
                if k == "sd_model":
                    copy_p.override_settings['sd_model_checkpoint'] = v
                else:
                    setattr(copy_p, k, v)

            if args.get("prompt") and p.prompt:
                if prompt_position == "start":
                    copy_p.prompt = args.get("prompt") + " " + p.prompt
                else:
                    copy_p.prompt = p.prompt + " " + args.get("prompt")

            if args.get("negative_prompt") and p.negative_prompt:
                if prompt_position == "start":
                    copy_p.negative_prompt = args.get("negative_prompt") + " " + p.negative_prompt
                else:
                    copy_p.negative_prompt = p.negative_prompt + " " + args.get("negative_prompt")

            proc = process_images(copy_p)
            images += proc.images

            if checkbox_iterate:
                p.seed = p.seed + (p.batch_size * p.n_iter)
            all_prompts += proc.all_prompts
            infotexts += proc.infotexts
        # end prompts_from_file.py
        ######################################################################################
        p.do_not_save_grid = True

        # Add the prompt from above
        p.prompt += StyleDict[poUseColor]
        
        PO_TO_CALL= self.check_and_install_potrace()

        proc = process_images(p)
        mixedImages = []

        try:
            # vectorize
            for i,img in enumerate(proc.images[::-1]): 
                if (not hasattr(img,"already_saved_as")) : continue
                fullfn = img.already_saved_as
                fullfnPath = pathlib.Path(fullfn)
                
                fullofpnm =  fullfnPath.with_suffix('.pnm') #for vectorizing

                fullofTPNG = fullfnPath.with_stem(fullfnPath.stem+ "_T")
                fullofTPNG = fullofTPNG.with_suffix('.png')

                fullof = pathlib.Path(fullfn).with_suffix('.'+poFormat)

                mixedImages.append([img,"PNG"])

                # set transparency to PNG, actually not vector feature, but people need it
                if poTransPNG:
                    self.doTransPNG(poTransPNGEps, mixedImages, img, fullofTPNG, poTransPNGQuant)

                if poDoVector:
                    self.doVector(poFormat, poOpaque, poTight, poKeepPnm, poThreshold, PO_TO_CALL, img, fullofpnm, fullof, mixedImages)

        except (Exception):
            raise Exception("VectorStudio: Execution of Potrace failed, check filesystem, permissions, installation or settings (is image saving on?)")

        # return Processed(p, mixedImages, p.seed, proc.info)
        # return Processed(p, images, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)
        return Processed(p, mixedImages, images, p.seed, proc.info, all_prompts=all_prompts, infotexts=infotexts)

    def doVector(self, poFormat, poOpaque, poTight, poKeepPnm, poThreshold, PO_TO_CALL, img, fullofpnm, fullof, mixedImages):
        # for vectorizing
        img.save(fullofpnm)
        print (pathlib.Path().resolve())
        args = [PO_TO_CALL,  "-b", poFormat, "-o", fullof, "--blacklevel", format(poThreshold, 'f')]
        if poOpaque: args.append("--opaque")
        if poTight: args.append("--tight")
        args.append(fullofpnm)

        p2 = subprocess.Popen(args)

        if not poKeepPnm:
            p2.wait()
            os.remove(fullofpnm)

        abspathsvg = os.path.abspath(fullof)
        mixedImages.append([abspathsvg,"SVG"]) # img, caption

    def doTransPNG(self, poTransPNGEps, mixedImages, img, fullofTPNG, poTransPNGQuant):
        #Image.quantize(colors=256, method=None, kmeans=0, palette=None)
        imgQ = img.quantize(colors=poTransPNGQuant, kmeans=0, palette=None)
        histo = imgQ.histogram()

        # get first pixel and assume it is background, best with Sticker style
        if (imgQ):
            bgI = imgQ.getpixel((0,0)) # return pal index
            bg = list(imgQ.palette.colors.keys())[bgI]

        E = poTransPNGEps # tolerance range if noisy

        imgT=imgQ.convert('RGBA')
        datas = imgT.getdata()
        newData = []
        for item in datas:
            if (item[0] > bg[0]-E and item[0] < bg[0]+E) and (item[1] > bg[1]-E and item[1] < bg[1]+E) and (item[2] > bg[2]-E and item[1] < bg[2]+E):
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)

        imgT.putdata(newData)
        imgT.save(fullofTPNG)
        mixedImages.append([imgQ,"PNG-quantized"])
        mixedImages.append([imgT,"PNG-transparent"])



def add_tab():
    haveControlnet = check_ext("controlnet")

    with gr.Blocks(analytics_enabled=False) as ui:

        with gr.Tab("Finetune PNG", elem_id="VS_FinetuneTab"):
            with gr.Column():
                gr.HTML(value=f"<iframe id=\"svgcode-iframe\" class=\"border-2 border-gray-200\" src=\"{iframesrcSVGCODE}\" title='description'></iframe>")

        with gr.Tab("Edit SVG", elem_id="VS_EditSVGTab"):
            with gr.Row(visible=haveControlnet):
                sendto_controlnet_txt2img = gr.Button("Send to txt2img ControlNet", visible=haveControlnet)
                sendto_controlnet_img2img = gr.Button("Send to img2img ControlNet", visible=haveControlnet)
                controlnet_max = opts.data.get("control_net_max_models_num", 1)
                sendto_controlnet_num = gr.Dropdown(list(range(controlnet_max)), label="ControlNet number", value="0", interactive=True, visible=(haveControlnet and controlnet_max > 1))

                sendto_controlnet_txt2img.click(
                            fn=None,
                            inputs=[sendto_controlnet_num],
                            outputs=[],
                            _js="vectorstudio_controlnet_send_txt2img"
                        )
                
                sendto_controlnet_img2img.click(
                    fn=None,
                    inputs=[sendto_controlnet_num],
                    outputs=[],
                    _js="vectorstudio_controlnet_send_img2img"
                )

            with gr.Column():
                gr.HTML(value=f"<iframe id=\"vectorstudio-iframe\" class=\"border-2 border-gray-200\" src=\"{iframesrc}\" title='description'></iframe>")

    return [(ui, "Vector Studio", "vector-studio")]


def after_component(component, **kwargs):
    global script_list_component

    # Add our buttons after each "send to extras" button
    if kwargs.get("elem_id") == "extras_tab":

            suffix = component.parent.elem_id

            if (suffix):
                with gr.Accordion("Vector Studio", open=False, elem_id="VectorStudio_ToolBox", visible=False):
                        with gr.Row():
                            edit_svg_button = gr.Button ("Edit SVG", elem_id="sendto_svgedit_button_"+suffix)
                            svgcode_button =  gr.Button ("Finetune PNG", elem_id="sendto_svgcode_button_"+suffix)
                            cycle_svg_bg_button  = gr.Button("Cycle BG", elem_id="svg_cycle_bg", visible=True)
                    
                            cycle_svg_bg_button.click(None,[],None,_js="vectorstudio_cycle_svg_bg")
                            edit_svg_button.click (None, [],None, _js="vectorstudio_send_gallery()" )
                            svgcode_button.click (None, [],None, _js="vectorstudio_send_gallery_svgcode()" )

    # get the dropdown component to depend on selected/active script.
    if kwargs.get("elem_id") == "script_list":
        script_list_component = component

script_callbacks.on_ui_tabs(add_tab)
script_callbacks.on_after_component(after_component)
