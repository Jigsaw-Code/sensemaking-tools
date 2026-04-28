<script>
	import "$styles/app.css";

	let { children } = $props();
</script>

<main id="content">
	{@render children?.()}
</main>
