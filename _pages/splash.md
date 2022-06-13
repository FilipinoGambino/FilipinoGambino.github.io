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
  caption: "Photo credit: [**Unsplash**](https://unsplash.com)"
excerpt: "Sphinx of black quartz, judge my vow."
intro: 
  - excerpt: 'This is an intro excpert'
feature_row:
  - image_path: assets/images/baseline_wandb.jpg
    alt: "placeholder image 1"
    title: "Placeholder 1"
    excerpt: "This is some sample content that goes here with **Markdown** formatting."
    url: "https://filipinogambino.github.io/ngorichs/combined_arms/combined-arms-part-1.html"
  - image_path: /assets/images/cnn.jpg
    image_caption: "Image courtesy of [Unsplash](https://unsplash.com/)"
    alt: "placeholder image 2"
    title: "Placeholder 2"
    excerpt: "This is some sample content that goes here with **Markdown** formatting."
    url: "(https://filipinogambino.github.io/ngorichs/combined_arms/combined-arms-part-2.html)"
  - image_path: /assets/images/cnn_emb_wandb.jpg
    title: "Placeholder 3"
    excerpt: "This is some sample content that goes here with **Markdown** formatting."
    url: "https://filipinogambino.github.io/ngorichs/combined_arms/combined-arms-part-3.html"
feature_row2:
  - image_path: /assets/images/mean_episode_length.JPG
    alt: "placeholder image 2"
    title: "Placeholder Image Left Aligned"
    excerpt: 'This is some sample content that goes here with **Markdown** formatting. Left aligned with `type="left"`'
    url: "#test-link"
    btn_label: "Read More"
    btn_class: "btn--primary"
feature_row3:
  - image_path: /assets/images/mean_episode_reward.JPG
    alt: "placeholder image 2"
    title: "Placeholder Image Right Aligned"
    excerpt: 'This is some sample content that goes here with **Markdown** formatting. Right aligned with `type="right"`'
    url: "#test-link"
    btn_label: "Read More"
    btn_class: "btn--primary"
feature_row4:
  - image_path: /assets/images/mean_training_reward.JPG
    alt: "placeholder image 2"
    title: "Placeholder Image Center Aligned"
    excerpt: 'This is some sample content that goes here with **Markdown** formatting. Centered with `type="center"`'
    url: "#test-link"
    btn_label: "Read More"
    btn_class: "btn--primary"
---

{% include feature_row id="intro" type="center" %}

{% include feature_row %}

{% include feature_row id="feature_row2" type="left" %}

{% include feature_row id="feature_row3" type="right" %}

{% include feature_row id="feature_row4" type="center" %}
