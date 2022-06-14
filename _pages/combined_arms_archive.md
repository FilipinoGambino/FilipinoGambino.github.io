---
layout: archive
permalink: /combined_arms/archive/
taxonomy: combined_arms
classes: wide
author_profile: false
sidebar:
  nav: "projects"
---

<style>
.center {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 50%;
} 
 
.column {
  float: left;
  width: 50%;
  padding: 5px;
}

.row::after {
  content: "";
  clear: both;
  display: table;
}
</style>

Hello and welcome to my first blog post! This is the start of a series of posts where I'm going to be working with the multi-agent [Combined Arms](https://www.pettingzoo.ml/magent/combined_arms) environment of the PettingZoo library and I'll keep adding parts, building it up, testing things out, and later implement a couple of papers I think are interesting.
<br />

## Purpose
The purpose of this series is to showcase a project to potential employers.

## The Environment
There are 2 teams contained in a 45x45 map and each team is composed of 45 melee units and 36 ranged units. Melee units have a shorter range for both attack and movement than their ranged counterparts, but have more health. The units or agents also slowly regenerate a small amount of their missing health since it takes multiple attacks to kill an agent. Agents are rewarded for injurying/killing opposing agents and negatively rewarded for both injuring/killing friendly agents or dying.

<div class="row">
  <div class="column">
    <img src="https://filipinogambino.github.io/ngorichs/assets/images/combined_arms_v6_opening.png" alt="Starting Position" height=350>
  </div>
  <div class="column">
    <img src="https://filipinogambino.github.io/ngorichs/assets/images/combined_arms_v6_one_step.png" alt="First Step" height=350>
  </div>
</div>
