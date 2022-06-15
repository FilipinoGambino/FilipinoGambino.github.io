---
layout: archive
permalink: /combined_arms/archive/
taxonomy: combined_arms
classes: wide
author_profile: false
sidebar:
  nav: "combined_arms"
---

<style>
.centertext {text-align: center;}

.center {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 80%;
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

Hello and welcome to my first blog post! This is the start of a series of posts where I'm going to be working with the multi-agent [Combined Arms](https://www.pettingzoo.ml/magent/combined_arms) environment of the PettingZoo library. I'll keep adding parts, building it up, testing things out, and later implement a couple of papers I think are interesting.
<br />

## Purpose
The purpose of this series is to showcase a project to potential employers.

## The Environment
There are 2 teams contained in a 45x45 map and each team is composed of 45 melee agents (red and green) and 36 ranged agents (blue and black). Melee agents have a shorter range for both attack and movement than their ranged counterparts, but they have more health. The agents also slowly regenerate a small amount of their missing health so cooperation is highly encouraged. Agents are rewarded for attacking or killing opposing agents (0.2 and 5 respectively) and negatively rewarded for attacking friendly agents or dying (-0.1 for both). Lastly, there's a small negative reward every step of -0.005 so that the agents don't learn to do nothing or just wander around. Last team standing, wins.

<div class="center">
  <table>
    <tr>
      <th></th>
      <th>Health</th>
      <th>Damage</th>
      <th>View Range</th>
      <th>Attack Range</th>
      <th>Move Range</th>
      <th>Health Regen</th>
    </tr>
    <tr>
      <td>Melee</td>
      <td style="text-align:center">10</td>
      <td style="text-align:center">2</td>
      <td style="text-align:center">6</td>
      <td style="text-align:center">1</td>
      <td style="text-align:center">1</td>
      <td style="text-align:center">0.1</td>
    </tr>
    <tr>
      <td>Ranged</td>
      <td style="text-align:center">3</td>
      <td style="text-align:center">2</td>
      <td style="text-align:center">6</td>
      <td style="text-align:center">2</td>
      <td style="text-align:center">2</td>
      <td style="text-align:center">0.1</td>
    </tr>
  </table>
</div>

<div class="row">
  <div class="column">
    <div class="centertext">Starting Positions</div>
    <img src="https://filipinogambino.github.io/ngorichs/assets/images/combined_arms_v6_opening.png" alt="Starting Position">
  </div>
  <div class="column">
    <div class="centertext">First Step</div>
    <img src="https://filipinogambino.github.io/ngorichs/assets/images/combined_arms_v6_one_step.png" alt="First Step">
  </div>
</div>
