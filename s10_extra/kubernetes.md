---
layout: default
title: Kubernetes
parent: S10 - Extra
nav_order: 7
nav_exclude: true
---

<img style="float: right;" src="../figures/icons/kubernetes.png" width="130">

# Kubernetes
{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

{: .warning }
> Module is still under development

## Kubernetes architechture

<p align="center">
  <img src="../figures/components_of_kubernetes.png" width="800">
  <br>
  <a href="https://kubernetes.io/docs/concepts/overview/components/"> Image credit </a>
</p>

## Minikube

### Exercises

1. Install [minikube](https://minikube.sigs.k8s.io/docs/start/)

2. Make sure that minikube is correctly installed by typing

   ```bash
   minikube
   ```

   in a terminal. Additionally, also check that [kubectl](https://kubernetes.io/docs/reference/kubectl/kubectl/) (the
   command line tool for kubernetes, its a dependency of minikube) is correctly installed by typing

   ```bash
   kubectl
   ```

   in a terminal.

## Yatai

[yatai](https://github.com/bentoml/Yatai)
