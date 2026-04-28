<script>
	import { onMount } from "svelte"; // @remove

// Qualtrics.SurveyEngine.addOnload(function() {
// });

// Qualtrics.SurveyEngine.addOnReady(function() {

	import Quotes from "$data/quotes.json";
	import Opinion from "$data/opinions.json";
	import Variables from "$data/variables.json"
	


	let chartEl;
    let modal;
    let elId = "chart"

	let groupedData = {};
	let opinionLookup = {};
	let viewMoreCount = 5;
	
	let colorMap = Variables.colors;

    let colorMapLight = Variables.colorsLight;

	let topicOrder = Variables.categoryOrder;

	const orderIndex = new Map(topicOrder.map((t, i) => [t, i]));
 

	function shuffle(array) {
		var m = array.length, t, i;

		while (m) {
			i = Math.floor(Math.random() * m--);

			t = array[m];
			array[m] = array[i];
			array[i] = t;
		}

		return array;
	}

	if (Quotes && Array.isArray(Quotes)) {
		const shuffledQuotes = shuffle([...Quotes]);

		shuffledQuotes.forEach(item => {
			const topic = item.topic;
			const opinion = item.opinion;
			
			if (!groupedData[topic]) {
				groupedData[topic] = {};
			}
			if (!groupedData[topic][opinion]) {
				groupedData[topic][opinion] = [];
			}
			groupedData[topic][opinion].push(item);
		});


		const sortedEntries = Object.entries(groupedData).sort((a, b) => {
			const ai = orderIndex.get(a[0]);
			const bi = orderIndex.get(b[0]);
			if (ai !== bi) return ai - bi;
		});
		groupedData = Object.fromEntries(sortedEntries);


	}

	function generateContent() {

        chartEl = document.getElementById(elId);
        modal = document.getElementById("modal-el");
        modal.querySelector('.modal-close').addEventListener('click', () => {
			modal.scrollTop = 0;
			modal.classList.remove('show-modal');
			viewMoreCount = 5;

            document.body.style.overflow = '';
            const viewMoreButton = modal.querySelector('.view-more');
            viewMoreButton.style.display = 'block';
			const viewMoreSheet = modal.querySelector('.view-more-sheet');
			viewMoreSheet.style.display = 'none';

        }); 

		var maxLength = Math.max.apply(null, Opinion.map(d => +d.count));
		Object.entries(groupedData).forEach(([topic, opinions], topicIndex) => {

			const sortedOpinions = Object.entries(opinions)
				.map(([opinion, items]) => {
					const count = opinionLookup[opinion].count;
					const gid = opinionLookup[opinion].gid;
					return [opinion, items, count, gid];
				})
				.sort((a, b) => b[2] - a[2]);

			const topicDiv = document.createElement('div');
			topicDiv.className = 'topic-group';
            topicDiv.style.setProperty('--color', '#' + colorMap[topicIndex]);

            const topicLabel = document.createElement('div');
            topicLabel.className = 'topic-label';
            topicLabel.style.backgroundColor = '#' + colorMapLight[topicIndex];
            topicLabel.style.border = '1px solid #' + colorMap[topicIndex];
            topicDiv.appendChild(topicLabel);

            const topicIcon = document.createElement('div');
            topicIcon.className = 'topic-icon';
			topicIcon.style.setProperty('--topic-icon-image', Variables.topicImages[topicIndex]);
			topicIcon.setAttribute("data-topic", 'topic-' + topicOrder.indexOf(topic));

            topicLabel.appendChild(topicIcon);


            const topicLabelP = document.createElement('p');
            topicLabelP.textContent = topic;
            topicLabelP.style.color = '#' + colorMap[topicIndex];
            topicLabel.appendChild(topicLabelP);

            const opinionWrapper = document.createElement('div');
            opinionWrapper.className = 'opinion-wrapper';
            opinionWrapper.style.setProperty('--color', '#' + colorMap[topicIndex]);
            topicDiv.appendChild(opinionWrapper);

			sortedOpinions.forEach(([opinion, items, count, gid], opinionIndex) => {
				const opinionGroupDiv = document.createElement('div');

				const isLast = opinionIndex === sortedOpinions.length - 1;
				const isLastBorder = opinionIndex === sortedOpinions.length - 2;
				const removeBorder = opinionIndex === sortedOpinions.length - 1;


                const isFirst = opinionIndex === 0;

				opinionGroupDiv.className = 'opinion-group';
				opinionGroupDiv.dataset.opinion = opinion.toLowerCase().replace(/ /g,"_");
				if(removeBorder) {
                    opinionGroupDiv.classList.add('remove-border');
                }
                if(isLastBorder) {
                    opinionGroupDiv.classList.add('last');
                }
                if(isFirst) {
                    opinionGroupDiv.classList.add('first');
                }
                
                const opinionBar = document.createElement('div');
                opinionBar.className = 'opinion-bar';
                opinionBar.style.setProperty('--color', '#' + colorMap[topicIndex]);
                if(isLast) {
                    opinionBar.classList.add('shortened');
                }
                if(isFirst) {
                    opinionBar.classList.add('shortened');
                }

                opinionGroupDiv.appendChild(opinionBar);

                const opinionBarData = document.createElement('div');
                opinionBarData.className = 'opinion-bar-data';
                opinionBarData.style.backgroundColor = '#' + colorMap[topicIndex];

				const opinionBarDataCount = document.createElement('div');
                opinionBarData.style.backgroundColor = '#' + colorMap[topicIndex];
				opinionBarDataCount.className = 'opinion-bar-data-count';
				opinionBarDataCount.textContent = count.toLocaleString();
				if (opinionIndex === 0) {
					opinionBarDataCount.textContent = count.toLocaleString() + " Quotes";
					opinionBarDataCount.classList.add("first-opinion");
				}

				let totalQuotes = count;

				opinionBarData.appendChild(opinionBarDataCount);

                opinionBar.appendChild(opinionBarData);
                const width = ((count / maxLength) * 90) + 10;
                opinionBarData.style.width = width + 'px';

                const opinionText = document.createElement('div');
                opinionText.className = 'opinion-text';
                opinionGroupDiv.appendChild(opinionText);

				const opinionLabelP = document.createElement('p');
				opinionLabelP.className = 'opinion-label';
				opinionLabelP.style.color = '#' + colorMap[topicIndex];
				opinionLabelP.textContent = 'Opinion';
                opinionLabelP.style.color = '#' + colorMap[topicIndex];
                opinionText.appendChild(opinionLabelP);

				const h3 = document.createElement('h3');
				h3.className = 'opinion-title';
				h3.innerHTML = opinion;
                h3.style.color = '#' + colorMap[topicIndex];
                opinionText.appendChild(h3);

                const opinionButton = document.createElement('button');
                opinionButton.className = 'opinion-button';
                opinionButton.textContent = 'View Quotes'
                opinionButton.style.backgroundColor = '#' + colorMapLight[topicIndex];
                opinionButton.style.color = '#' + colorMap[topicIndex];
                opinionGroupDiv.appendChild(opinionButton);
                opinionButton.addEventListener('click', () => {

                    modal.classList.add('show-modal');

					let quotes = items.slice(0,20);
					
                    modal.querySelector('.topic-label-top').style.color = '#' + colorMap[topicIndex];
                    modal.querySelector('.topic-label').style.color = '#' + colorMap[topicIndex];
                    modal.querySelector('.topic-label').textContent = opinion;
                    modal.querySelector('.topic-count').innerHTML = 'SUPPORTING QUOTES (' + quotes.length + '):<span><i>We used AI to help us find personal stories, differing outlooks, and questions from each participant. These 20 quotes were then randomly selected from those responses.</i></span><span><i>Some of these quotes have been shortened for clarity.</i></span>';

                    modal.querySelector('.topic').innerHTML = quotes.filter((d,i) => {
						return i < viewMoreCount;
					}).map(q => {
								return '<li class="quote">“' + q.quote + '”</li>';
					}).join('');

                    let viewMoreButton = modal.querySelector('.view-more');
					let newViewMoreButton = viewMoreButton.cloneNode(true);
					viewMoreButton.parentNode.replaceChild(newViewMoreButton, viewMoreButton);
					viewMoreButton = newViewMoreButton;

					const viewMoreLink = modal.querySelector('.view-more-sheet').querySelector('span');
					const viewMoreHref = modal.querySelector('.view-more-sheet').querySelector('a');
					if(Variables.googleSheetsLink !== ""){
						viewMoreHref.href = Variables.googleSheetsLink + gid;
					}
					else {
						modal.querySelector('.view-more-sheet').style.visibility = "hidden";
					}
					viewMoreLink.textContent = totalQuotes.toLocaleString();
                    viewMoreButton.style.backgroundColor = '#' + colorMapLight[topicIndex];
                    viewMoreButton.style.color = '#' + colorMap[topicIndex];

                    viewMoreButton.addEventListener('click', () => {
						viewMoreCount = viewMoreCount + 5;
                        modal.querySelector('.topic').innerHTML = quotes
							.filter((d,i) => {
								return i < viewMoreCount;
							})
							.map((q) => {
										return '<li class="quote">“' + q.quote + '”</li>';
							}).join('');
							if(viewMoreCount > quotes.length - 1) {
								viewMoreButton.style.display = 'none';
								let viewMoreLink = modal.querySelector('.view-more-sheet')
								viewMoreLink.style.display = 'block';
							}						
                    });

                    document.body.style.overflow = 'hidden';
                });

                opinionWrapper.appendChild(opinionGroupDiv);
			});

            chartEl.appendChild(topicDiv);

		});
	}

	onMount(() => { // @remove

		opinionLookup = Opinion.reduce((map, item) => {
			map[item.opinion] = item;
			return map;
		}, {});

		generateContent();

        // document.querySelector("label").htmlFor = "idOfAssociatedInput";

	});  // @remove
// })

// Qualtrics.SurveyEngine.addOnUnload(function() {

// })


</script>

<svelte:boundary>
	<div id="wrapper">
		<div class="svg-container">
			<div class="svg-wrapper" id="chart" style="width: 100%; max-width: 750px; margin: auto;">
			</div>
		</div>
		<div id="modal-el" class="modal list-modal">
			<div class="modal-content">
				<button class="modal-close" aria-label="Close">×</button>
				<p class="topic-label-top">OPINION:</p>
				<p class="topic-label"></p>
				<p class="topic-count"></p>
				<ul class="topic"></ul>
				<button class="view-more">View more quotes <span style="transform: rotate(45deg) scale(0.6) translate(0, -4px); display: inline-block;">◢</span></button>
				<p class="view-more-sheet">You can <a href="/" target="_blank">view you all <span>X</span> quotes here</a>. This link will open a new page.</p>
			</div>
		</div> 
	</div>
</svelte:boundary>

