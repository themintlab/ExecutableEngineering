# Textbook design

Many numerical methods explored here are mature, having benefited from decades of development. However, their efficient and effective use involves subtle nuances. The primary goal of this text is to develop a solid conceptual understanding of these methods. 

Upon completion, students should be able to:
- Break down complex problems into manageable components.
- Make informed choices of suitable tools and critique default recommendations.
- Interpret solutions and evaluate metrics associated with the solution process.

## Policy on Generative AI
It is the opinion of the author that GenAI's capability to deliver fast, precise answers mimics having a bright classmate or access to a previous year's solutions.Beyond the obvious fact that neither are infalible, short-cutting the learning process by copying the answers is *fraudulent* and not conducive to development of a conceptual understanding.

However, functioning effectively in an engineering team is a key professional skill. In this context, GenAI (or to a lesser extent pacakge tools, online discussion boards, other students)can act as a capable team member that accelerates laborious tasks. This allows you, as the lead engineer, to focus on interpreting solutions. When used responsibly, these tools serve as a "cognitive elevator," accelerating learning past the foundational "Remember" and "Understand" stages of Bloom's Taxonomy directly into "Analyze," "Evaluate," and "Create." This frees up cognitive load, allowing you to focus on mastering core concepts and practical implementations.

## Design Elements

### Open-Source, Accessible Tools
This text is designed to be conveniently accessible through a modern web browser, introducing tools you will likely always have access to:
- Built in a [GitHub](https://github.com/) repository and hosted on [GitHub Pages](https://pages.github.com/).
- Formatted as a collection of Jupyter Notebooks assembled with [Jupyter Book](https://jupyterbook.org/).
- Live, in-browser execution provided by the [Binder](https://mybinder.org/) project.
- Can be launched and viewed directly in [Google Colab](https://colab.google.com/).
- Relies on [Python](https://www.python.org/), specifically the [`numpy`](https://numpy.org/) and [`scipy`](https://scipy.org/) packages.

### Experiential Learning and Interactivity
Studies consistently show the benefits of experiential learning. This text leverages live, in-browser execution to facilitate this:

- **Interactivity:** The provided code is editable (see note below), allowing students to explore and experiment conveniently.
- **Constructive Friction:** Code is intentionally presented un-executed. Students are encouraged to predict outcomes before running it. This helps dispel the [illusion of competence](https://www.coursera.org/articles/illusion-of-competence) and builds essential critical thinking skills.
- **Spaced Repetition:** Numerical methods are hierarchical. As tools become more sophisticated, students will naturally revisit and reinforce fundamental concepts.

## Live Execution Instructions

Executing code requires a computing environment known as a **kernel**. Your browser acts as the front-end, while all computations occur in the backend kernel.

### In-textbook execution
On the top-right hand side of the page, you should see a 'power button' ![alt text](../../images/power_button.png). Clicking it will launch a jupyter kernel and enable execution. The top of the page will now have a bar that performs:

- Run all cells
- Reset notebook and restart kernel
- Clear all (output) cells
- Launch notebook in Jupyter (a differnt view of the same)

Each cell in the text will be endowed with two button:
- Run cell
- Clear cell outputs

> Note that the kernel is shared between all the cells. If you assign a variable in one, it will persist to the others. Similarly, any import statements must be run before the cells that use them. 

This feature is under development by the generous contribution of the open-source community. It may not always work as expected.

### Google Colab
[Google Colab](https://colab.research.google.com/) is a free, cloud based jupyter notebook environment. Clicking on the badge in the upper left corner of the page will open the notebook in colab. You will need to be logged into your google account to use it. It has Gemini built in a full set of scientific tools. 

The notebooks are designed to be presented in the Colab 'slideshow' mode. 

As a commercial product Google Colab is typically more robust than the in-browser execution, however it should be noted that their normal terms of use (including privacy) apply to colab and Gemini usage.