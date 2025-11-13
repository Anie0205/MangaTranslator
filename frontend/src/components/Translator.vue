<template>
  <div class="translator">
    <h2>Manga Translator</h2>

    <form @submit.prevent="handleTranslate">
      <input type="file" @change="handleFileChange" required />
      
      <label>Source Language:</label>
      <select v-model="sourceLang">
        <option value="japanese">Japanese</option>
        <option value="korean">Korean</option>
        <option value="chinese">Chinese</option>
      </select>

      <label>Target Language:</label>
      <input v-model="targetLang" placeholder="e.g. english" />

      <button type="submit">Translate</button>
    </form>

    <div v-if="loading">Translating...</div>

    <div v-if="result">
      <h3>OCR Extracted Text:</h3>
      <pre>{{ result.original_text }}</pre>

      <h3>Translated Text:</h3>
      <pre>{{ result.translated_text }}</pre>

      <h3>Translated Image:</h3>
      <img :src="'data:image/png;base64,' + result.image_base64" style="max-width: 100%;" />
    </div>

  </div>
</template>

<script>
export default {
  data() {
    return {
      file: null,
      sourceLang: "japanese",
      targetLang: "english",
      result: null,
      loading: false,
    };
  },
  methods: {
    handleFileChange(event) {
      this.file = event.target.files[0];
    },
    async handleTranslate() {
      if (!this.file) return;

      this.loading = true;
      const formData = new FormData();
      formData.append("file", this.file);
      formData.append("source_lang", this.sourceLang);
      formData.append("target_lang", this.targetLang);

      try {
        const res = await fetch("http://localhost:8000/translate/", {
          method: "POST",
          body: formData,
        });
        const data = await res.json();
        this.result = data;
      } catch (err) {
        alert("Error during translation");
      } finally {
        this.loading = false;
      }
    },
  },
};
</script>

<style scoped>
.translator {
  max-width: 600px;
  margin: auto;
  padding: 20px;
}
pre {
  background: #f4f4f4;
  padding: 10px;
  white-space: pre-wrap;
}
</style>
