document.getElementById("query-form").addEventListener("submit", async function (e) {
    e.preventDefault();

    const formData = new FormData(this);

    try {
        // Send the form data to the server
        const response = await fetch('/search', {
            method: 'POST',
            body: formData
        });

        // Check if the response is okay
        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }

        const results = await response.json();
        const resultsContainer = document.getElementById("results");
        resultsContainer.innerHTML = ""; // Clear previous results

        // Render each result
        results.forEach(({ filename, score }) => {
            const container = document.createElement("div");
            const img = document.createElement("img");
            const caption = document.createElement("p");

            img.src = `/static/${filename}`;
            caption.innerText = `Score: ${score.toFixed(2)}`;

            container.appendChild(img);
            container.appendChild(caption);
            resultsContainer.appendChild(container);
        });
    } catch (error) {
        console.error("Error fetching search results:", error);
        alert("There was an error fetching search results. Please try again.");
    }
});
