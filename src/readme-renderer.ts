// Import the marked library
import { marked } from 'marked';

// Function to fetch and render README.md content
export async function loadAndRenderReadme(): Promise<void> {
    try {
        // Fetch the README.md file
        const response = await fetch('/README.md');

        if (!response.ok) {
            throw new Error(`Failed to load README.md: ${response.status} ${response.statusText}`);
        }

        // Get the markdown content
        const markdown = await response.text();

        // Convert markdown to HTML
        const html = marked.parse(markdown) as string;

        // Get the readme content element
        const readmeElement = document.getElementById('readme-content');

        if (readmeElement) {
            // Set the HTML content
            readmeElement.innerHTML = html;
        } else {
            console.error('README content element not found');
        }
    } catch (error) {
        console.error('Error loading or rendering README:', error);
    }
}
