---
title: "Splash"
layout: splash
permalink: /splash/
date: 2022-03-23T11:48:41-04:00
header:
  overlay_color: "#000"
  overlay_filter: "0.5"
  overlay_image: /assets/images/lr_schedule_plot.jpg
  actions:
    - label: "Download"
      url: "https://filipinogambino.github.io/ngorichs/"
  caption: "Photo credit: [**Me**](https://filipinogambino.github.io/ngorichs/)"
excerpt: "Sphinx of black quartz, judge my vow."
intro: 
  - excerpt: 'This is an intro excerpt'
gallery:
  - url: "https://filipinogambino.github.io/ngorichs/combined_arms/combined-arms-part-1.html"
    image_path: assets/images/baseline_wandb.jpg
    alt: "placeholder image 1"
    title: "Image 1 title caption"
  - url: "https://filipinogambino.github.io/ngorichs/combined_arms/combined-arms-part-2.html"
    image_path: /assets/images/cnn.jpg
    alt: "placeholder image 2"
    title: "Image 2 title caption"
  - url: "https://filipinogambino.github.io/ngorichs/combined_arms/combined-arms-part-3.html"
    image_path: /assets/images/cnn_emb_wandb.jpg
    alt: "placeholder image 3"
    title: "Image 3 title caption"
feature_row:
  - image_path: assets/images/baseline_wandb.jpg
    alt: "placeholder image 1"
    title: "Placeholder 1"
    excerpt: "This is some sample content that goes here with **Markdown** formatting."
    url: "https://filipinogambino.github.io/ngorichs/combined_arms/combined-arms-part-1.html"
    btn_label: "Read More"
    btn_class: "btn--primary"
  - image_path: /assets/images/cnn.jpg
    image_caption: "Image courtesy of [Me](https://filipinogambino.github.io/ngorichs/)"
    alt: "placeholder image 2"
    title: "Placeholder 2"
    excerpt: "This is some sample content that goes here with **Markdown** formatting."
    url: "https://filipinogambino.github.io/ngorichs/combined_arms/combined-arms-part-2.html"
    btn_label: "Read More"
    btn_class: "btn--primary"
  - image_path: /assets/images/cnn_emb_wandb.jpg
    title: "Placeholder 3"
    excerpt: "This is some sample content that goes here with **Markdown** formatting."
    url: "https://filipinogambino.github.io/ngorichs/combined_arms/combined-arms-part-3.html"
    btn_label: "Read More"
    btn_class: "btn--primary"
---

{% include feature_row id="intro" type="center" %}

{% include gallery caption="This is a sample gallery with **Markdown support**." %}

{% include feature_row %}
