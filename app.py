#conda env: insurtech
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import streamlit as st
import io
from PIL import Image
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb
import cv2 as cv

 
def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a numpy array
    
    """
    # draw the figure first
    fig.canvas.draw()
 
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

# Convert the numpy array to a binary object in memory
def numpy_to_binary(arr):
    is_success, buffer = cv.imencode(".jpg", arr)
    io_buf = io.BytesIO(buffer)
    print(type(io_buf))
    return io_buf.read()

st.title('WordCloud Maker')

def clean_text(text):
    
    # remove newlines
    text = text.replace('\n', ' ')
    #Remove RT
    text = re.sub(r'RT', '', text)
    
    #Fix &
    text = re.sub(r'&amp;', '&', text)
    
    #Remove punctuations
    text = re.sub(r'[?!.;:,#@-]', '', text)
    text = re.sub(r'😒😒😒, 😇, 🕺', '', text)
    text = re.sub(r'<media', '', text)
    text = re.sub(r'omitted>', '', text)
    


    #Convert to lowercase to maintain consistency
    text = text.lower()
    return text

def remove_emoji(string):
    emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002500-\U00002BEF"  # chinese char
                            u"\U00002702-\U000027B0"
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            u"\U0001f926-\U0001f937"
                            u"\U00010000-\U0010ffff"
                            u"\u2640-\u2642"
                            u"\u2600-\u2B55"
                            u"\u200d"
                            u"\u23cf"
                            u"\u23e9"
                            u"\u231a"
                            u"\ufe0f"  # dingbats
                            u"\u3030"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def gen_freq(text):
    #splitting the text
    text = text.split(' ')

    # collapsing numerous white spaces
    def containsNonAscii(s):
        return any(ord(i)>127 for i in s)
    text = [word for word in text if not containsNonAscii(word)]

    text = [word for word in text if word.isalnum()]
    #Create word frequencies using word_list
    word_freq = pd.Series(text).value_counts()
    
    return word_freq


def inputs():
    #inputs
    file = st.file_uploader('Upload .txt file', type='.txt', accept_multiple_files=False, key=None, help=None, 
                                on_change=None, args=None, kwargs=None)

    with st.sidebar:
        st.subheader("Configure the WordCloud")                       

        width= st.number_input('Width of the output', min_value=150, max_value=None, value=550, step=None, 
                            format=None, key=None, help=None, on_change=None, args=None, kwargs=None)

        height= st.number_input('Height of the output', min_value=100, max_value=None, value=330, step=None, 
                            format=None, key=None, help=None, on_change=None, args=None, kwargs=None)
        max_words= st.slider('Set number of Words', min_value=4, max_value=100, value=20, step=None, format=None, 
                            key=None, help=None, on_change=None, args=None, kwargs=None)

        background_color= st.color_picker('Background Color, default is black', value=None, key=None, help=None, 
                                            on_change=None, args=None, kwargs=None)
        colormap = "plasma"

    #actions
    run = st.button('Generate Cloud', key=None, help=None, on_click=None, args=None, kwargs=None)
    return file, width, height, max_words, background_color, run


#footer
def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 105px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        " with ❤️ by ",
        link("https://twitter.com/AbrahamOwos", "@AbrahamOwos"),
        br(),
        " ⚙️ Want me to change or add someing? Hit me up" ,
    ]
    layout(*myargs)



if __name__ == '__main__':
    file, width, height, max_words, background_color, run = inputs()
    footer()
    if run == True:
        if file is not None:
            file = stringio = io.StringIO(file.getvalue().decode("utf-8"))
            file = file.read()
            text = clean_text(file)
            text = remove_emoji(text)
            word_freq = gen_freq(text)
            #converting the word to string
            word_freq.index =word_freq.index.astype("object")
            word_freq = word_freq.drop(labels= STOPWORDS, errors= 'ignore')

            wdc = WordCloud(width=width, height=height, max_words=max_words, background_color=background_color, 
                            min_word_length = 4, colormap = "plasma").generate_from_frequencies(word_freq)

            
            fig, ax = plt.subplots()
            plt.imshow(wdc, interpolation='bilinear')
            plt.axis('off')
            
            im = numpy_to_binary(fig2data ( fig ))
            st.pyplot(fig)

            download = st.download_button('Download Image', im, file_name='wordcloud.png', mime='image/png', key=None, help=None, 
                              on_click=None, args=None, kwargs=None)
        else:
            st.write('Please upload the document first')