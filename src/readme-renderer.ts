// Import the marked library
import { marked } from 'marked';

// Get the base URL from the import.meta.env
const BASE_URL = import.meta.env.BASE_URL || '/';

// Function to fetch and render README.md content
export async function loadAndRenderReadme(): Promise<void> {
    try {
        // Fetch the README.md file with the correct base path
        const response = await fetch(`${BASE_URL}README.md`);

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
