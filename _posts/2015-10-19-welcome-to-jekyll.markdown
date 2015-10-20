---
layout: post
comments: true
title:  "How to make a blog like this"
date:   2015-10-19 16:46:27
categories: jekyll update
mathjax: true
---

## Motivation
I have been seeing beautiful blog posts coupled with great contents such as 

* [The Zen of Gradient Descent](
    http://blog.mrtz.org/2013/09/07/the-zen-of-gradient-descent.html)
* [The Unreasonable Effectiveness of Recurrent Neural Networks]( 
    http://karpathy.github.io/2015/05/21/rnn-effectiveness/ )
* [Understanding LSTM Networks]( 
    http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

My definition of being "beautiful" is just text and no distraction, look
good both on mobile and on desktop, and display Latex. My definition of "great
content" often involve machine learning, optimization, and related topics. 

## Setting up
So without being further ado, let's start with how to make a blog like those
three mentioned above blog. You will need a couple components:

* [Jekyll](http://jekyllrb.com): Pretty easy to install on Mac, just follow the
  link
* [Github Pages](https://pages.gith…): Also pretty easy to setup. Create a
  github account if you haven't got one, then follow the link.

After spending some good hours and failing at setting up the various 
requirements, here is the trick I found: just clone the blog of the people
above. In particular, I clone Karpathy blog into mine. Many thanks to Andre
Karpathy for this, and my apology if my blog accidentally contains some stuffs
from yours. Ok now open **Terminal** in Mac:

```
mkdir MyBlog
# Downloading sample github page from Andre Karpathy
git clone https://github.com/karpathy/karpathy.github.io.git
# Downloading your own github page blog. Replacing username with your github
# user name
git clone https://github.com/username/username.github.io.git
cp ./karpathy.github.io/* ./username.github.io/
# Removing stuff associated with the original owner
rm nntutorial.md
rm _posts/* # Hist posts
```

You will need to change a few more information. Open `_config.yaml`\_ and change
all the detail in there to your username. Change things in `about.md` as well. 
Also in `_layouts/post.html` and
`_layouts/page.html`, there is a part with "karpathy", change that to your
username. Basically, search for all text with the original owner's name and
replace them with your name. You might also want to delete everything in i
`assets`, as those are the picture that the original owner uses.  

And voila, you are done.

## Hello World
To create your first blog post, again copy over a sample post say from

```
cp ./karpathy.github.io/_posts/2015-05-21-rnn-effectiveness.markdown
   ./username.github.io/_posts/
```
And you can start changing things in there. Then do this `jekyll serve` inside
your `username.github.io` folder. You should see something like this if
successful 

```
Configuration file: /Users/hd/Documents/Blog/hduongtrong.github.io/_config.yml
            Source: /Users/hd/Documents/Blog/hduongtrong.github.io
       Destination: /Users/hd/Documents/Blog/hduongtrong.github.io/_site
      Generating... 
     Build Warning: Layout 'none' requested in feed.xml does not exist.
                    done.
 Auto-regeneration: enabled for '/Users/hd/Documents/Blog/hduongtrong.github.io'
Configuration file: /Users/hd/Documents/Blog/hduongtrong.github.io/_config.yml
    Server address: http://127.0.0.1:4000/
  Server running... press ctrl-c to stop.
```

Open your web browser, and go to `http://127.0.0.1:4000/`, you will see your
post there. 

Now to push this online, just 

```
cd username.github.io/
git add .
git commit -m "First blog"
git push origin master
```
And go to `username.github.io`.
